# SPDX-License-Identifier: Apache-2.0
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import torch.distributed as dist
import torch_npu
from torchtitan.config import derive, override
from torchtitan.observability.profiler import Profiler

logger = logging.getLogger(__name__)


class NPUProfiler(Profiler):
    """Collect profiling data for CPU/NPU devices."""

    @dataclass(kw_only=True, slots=True)
    class Config(Profiler.Config):
        start_step: int | None = None
        end_step: int | None = None
        save_dir: str | None = None
        level: str = "level1"
        aic_metrics_type: str = "PipeUtilization"
        with_cpu: bool = True
        analyse_flag: bool = True
        data_simplification: bool = False
        record_shapes: bool = False
        with_stack: bool = False
        profile_memory: bool = False
        ranks: list[int] | None = field(default_factory=lambda: [0])

        def __post_init__(self) -> None:
            if self.start_step is None and self.end_step is None:
                Profiler.Config.__post_init__(self)
            elif self.start_step is None or self.end_step is None:
                raise ValueError("start_step and end_step must be provided together")
            elif self.start_step < 0 or self.end_step <= self.start_step:
                raise ValueError("profiler requires 0 <= start_step < end_step")
            if self.level not in ("level0", "level1", "level2"):
                raise ValueError(
                    f"profiler level must be level0, level1 or level2, got {self.level!r}"
                )
            if self.aic_metrics_type not in (
                "PipeUtilization",
                "ArithmeticUtilization",
            ):
                raise ValueError(
                    "aic_metrics_type must be PipeUtilization or ArithmeticUtilization, "
                    f"got {self.aic_metrics_type!r}"
                )
            if (
                self.ranks is not None
                and self.ranks != [-1]
                and any(rank < 0 for rank in self.ranks)
            ):
                raise ValueError(
                    "ranks must contain nonnegative global ranks, or be [-1] or None for all ranks"
                )
            if cast(Profiler.Config, self).cuspy is not None:
                raise ValueError("NPU profiling cannot be combined with cuspy")

    def build_torch_profiler(
        self, *, global_step: int, base_folder: str, leaf_folder: str
    ):
        """Start a periodic schedule or a relative one-shot profiling window."""
        cfg = cast(Profiler, self)._config
        if not cfg.enable_profiling:
            return None

        rank = dist.get_rank() if dist.is_initialized() else 0
        if cfg.ranks is not None and cfg.ranks != [-1] and rank not in cfg.ranks:
            return None

        trace_dir = (
            os.path.join(base_folder, cfg.save_traces_folder)
            if cfg.save_dir is None
            else cfg.save_dir
        )

        # Use the recipe's periodic schedule when no explicit window is provided.
        if cfg.start_step is None:
            warmup, active = cfg.profiler_warmup, cfg.profiler_active
            wait = cfg.profile_freq - (warmup + active)
            additional_params = {
                key: value
                for key, value in [
                    ("repeat", cfg.profiler_repeat),
                    ("skip_first", cfg.profiler_skip_first),
                    ("skip_first_wait", cfg.profiler_skip_first_wait),
                ]
                if value is not None
            }
            logger.info(
                "Profiling every %d steps from restored step %d on rank %d; "
                "traces will be saved at %s",
                cfg.profile_freq,
                global_step,
                rank,
                trace_dir,
            )
        else:
            wait, warmup = 0, 0
            active = cfg.end_step - cfg.start_step
            additional_params = {"repeat": 1, "skip_first": cfg.start_step}
            logger.info(
                "Profiling global steps (%d, %d] on rank %d; traces will be saved at %s",
                global_step + cfg.start_step,
                global_step + cfg.end_step,
                rank,
                trace_dir,
            )

        activities = [torch_npu.profiler.ProfilerActivity.NPU]
        if cfg.with_cpu:
            activities.append(torch_npu.profiler.ProfilerActivity.CPU)

        profiler_level = {
            "level0": torch_npu.profiler.ProfilerLevel.Level0,
            "level1": torch_npu.profiler.ProfilerLevel.Level1,
            "level2": torch_npu.profiler.ProfilerLevel.Level2,
        }[cfg.level]

        aic_metrics = {
            "PipeUtilization": torch_npu.profiler.AiCMetrics.PipeUtilization,
            "ArithmeticUtilization": torch_npu.profiler.AiCMetrics.ArithmeticUtilization,
        }[cfg.aic_metrics_type]

        profile_schedule = torch_npu.profiler.schedule(
            wait=wait, warmup=warmup, active=active, **additional_params
        )
        # Align the initial schedule lookup with the restored global step.
        if cfg.start_step is None and global_step != 0:
            original_schedule = profile_schedule

            def resumed_schedule(step: int):
                # Construction evaluates step 0 before the global counter is restored.
                return original_schedule(global_step if step == 0 else step)

            profile_schedule = resumed_schedule

        prof = torch_npu.profiler.profile(
            activities=activities,
            schedule=profile_schedule,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                trace_dir, worker_name=f"rank{rank}", analyse_flag=cfg.analyse_flag
            ),
            record_shapes=cfg.record_shapes,
            with_stack=cfg.with_stack,
            profile_memory=cfg.profile_memory,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=profiler_level,
                aic_metrics=aic_metrics,
                data_simplification=cfg.data_simplification,
            ),
        )

        if cfg.start_step is None:
            prof.step_num = global_step
        prof.__enter__()
        return prof


@override(
    target=Profiler.Config, exact=True, description="Profiling support for NPU devices"
)
def npu_profiler(
    cfg: Profiler.Config,
    *,
    enable_profiling: bool | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    level: str = "level1",
    aic_metrics_type: str = "PipeUtilization",
    with_cpu: bool = True,
    analyse_flag: bool = True,
    data_simplification: bool = False,
    record_shapes: bool = False,
    with_stack: bool = False,
    profile_memory: bool = False,
    ranks: Sequence[int] | None = (0,),
    save_dir: str | None = None,
) -> NPUProfiler.Config:
    """Configure CPU/NPU profiling with an optional relative one-shot window.

    By default, preserve the original periodic schedule and follow global training
    progress across resumes. Providing both window bounds selects one recording
    cycle without warmup, relative to the start of this run, including a resume.

    Args:
        cfg: Original profiler configuration, including memory snapshot settings.
        enable_profiling: Enable trace collection; None preserves the original setting.
        start_step: Steps to skip in window mode; provide together with end_step.
        end_step: Last relative step to record (inclusive); provide with start_step.
        level: Profiling detail level: level0, level1 or level2; defaults to level1.
        aic_metrics_type: AI Core metrics: PipeUtilization or ArithmeticUtilization.
        with_cpu: Record CPU events alongside NPU events; enabled by default.
        analyse_flag: Automatically analyse captured data; enabled by default.
        data_simplification: Enable native profiling data simplification; defaults to False.
        record_shapes: Record operator input shapes.
        with_stack: Record operator call stacks.
        profile_memory: Record profiler memory events, separate from snapshots.
        ranks: Global ranks to collect; [-1] or None collects all ranks.
        save_dir: Explicit output directory, used directly. None resolves the original
            save_traces_folder relative to the training output directory.

    Returns:
        Configuration that builds NPU Profiler.
    """
    return derive(
        cfg,
        NPUProfiler.Config,
        enable_profiling=(
            cfg.enable_profiling if enable_profiling is None else enable_profiling
        ),
        start_step=start_step,
        end_step=end_step,
        level=level,
        aic_metrics_type=aic_metrics_type,
        with_cpu=with_cpu,
        analyse_flag=analyse_flag,
        data_simplification=data_simplification,
        record_shapes=record_shapes,
        with_stack=with_stack,
        profile_memory=profile_memory,
        ranks=None if ranks is None else list(ranks),
        save_dir=save_dir,
    )
