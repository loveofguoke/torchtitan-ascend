# SPDX-License-Identifier: Apache-2.0
"""CPU regressions using real TorchTitan configs and PyTorch schedules.

Run with ``python -m pytest --noconftest tests/unit_tests/cpu/test_profiler.py``
in an environment matching the TorchTitan baseline. The NPU backend and runtime rank
queries are mocked; these tests do not validate device traces or snapshots.
"""

import importlib
import json
import os
import sys
from dataclasses import dataclass, field, fields
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch.profiler import ProfilerAction
from torchtitan.config import Configurable
from torchtitan.config.override import (
    apply_overrides,
    OverrideConfig,
    parse_cli_imports,
)
from torchtitan.observability.profiler import CuspyProfilerConfig, Profiler

TARGET = "torchtitan_ascend.overrides.profiler.npu_profiler"
MODULE = "torchtitan_ascend.overrides.profiler"


def make_profiler_config(**overrides):
    # Keep public test inputs independent of TorchTitan's defaults.
    options = dict(
        enable_profiling=False,
        profile_freq=10,
        profiler_warmup=3,
        profiler_active=1,
        profiler_repeat=None,
        profiler_skip_first=None,
        profiler_skip_first_wait=None,
        save_traces_folder="test-traces",
        enable_memory_snapshot=False,
        memory_snapshot_freq=None,
        save_memory_snapshot_folder="test-snapshots",
        memory_snapshot_max_entries=1234,
        cuspy=None,
    )
    options.update(overrides)
    return Profiler.Config(**options)


class FakeProfile:
    """Model backend counter updates without collecting device events."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.step_num = 0
        self.current_action = kwargs["schedule"](0)
        self.events = []

    def __enter__(self):
        self.events.append(("enter", self.step_num, self.current_action))
        return self

    def step(self):
        self.events.append(("step", self.step_num, self.current_action))
        self.step_num += 1
        self.current_action = self.kwargs["schedule"](self.step_num)

    def __exit__(self, *exc):
        self.events.append(("exit", *exc))
        return False


def fake_npu_module():
    def schedule(**kwargs):
        if kwargs["warmup"] == 0:
            with pytest.warns(UserWarning, match="warmup"):
                return torch.profiler.schedule(**kwargs)
        return torch.profiler.schedule(**kwargs)

    module = ModuleType("torch_npu")
    module.__dict__["profiler"] = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU="CPU", NPU="NPU"),
        ProfilerLevel=SimpleNamespace(
            Level0="level0", Level1="level1", Level2="level2"
        ),
        AiCMetrics=SimpleNamespace(
            PipeUtilization="PipeUtilization",
            ArithmeticUtilization="ArithmeticUtilization",
        ),
        schedule=Mock(side_effect=schedule),
        profile=Mock(side_effect=FakeProfile),
        tensorboard_trace_handler=Mock(return_value=object()),
        _ExperimentalConfig=Mock(side_effect=lambda **kwargs: kwargs),
    )
    return module


@pytest.fixture(scope="module")
def profiler_module():
    parent = importlib.import_module("torchtitan_ascend.overrides")
    overrides = importlib.import_module("torchtitan.config.override")
    # Restore both import state and registration if another test imported it first.
    with pytest.MonkeyPatch.context() as mp, patch.dict(overrides._REGISTRY):
        overrides._REGISTRY.pop(TARGET, None)
        mp.setitem(sys.modules, "torch_npu", fake_npu_module())
        # A placeholder records the previous value before import replaces it.
        mp.setitem(sys.modules, MODULE, None)
        del sys.modules[MODULE]
        mp.setattr(parent, "profiler", None, raising=False)
        yield importlib.import_module(MODULE)


@pytest.fixture
def runtime(profiler_module, monkeypatch):
    npu = fake_npu_module()
    monkeypatch.setattr(profiler_module, "torch_npu", npu)
    initialized = Mock(return_value=True)
    rank = Mock(return_value=0)
    monkeypatch.setattr(torch.distributed, "is_initialized", initialized)
    monkeypatch.setattr(torch.distributed, "get_rank", rank)

    def build(
        *,
        original=None,
        global_step=0,
        base_folder="base",
        leaf_folder="leaf",
        **options
    ):
        config = profiler_module.npu_profiler(
            make_profiler_config() if original is None else original,
            enable_profiling=True,
            **options,
        )
        return config.build().build_torch_profiler(
            global_step=global_step, base_folder=base_folder, leaf_folder=leaf_folder
        )

    return SimpleNamespace(
        npu=npu.profiler, rank=rank, initialized=initialized, build=build
    )


class ProfilerRoot(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        profiler: Profiler.Config = field(default_factory=make_profiler_config)


def training_actions(prof, count, *, restored_step=0):
    actions = []
    for index in range(count):
        # Training step n uses the current action before completion calls step().
        actions.append((restored_step + index + 1, prof.current_action))
        prof.step()
    return actions


def recorded_steps(actions):
    return [
        step
        for step, action in actions
        if action in (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE)
    ]


def test_factory_preserves_public_fields_without_mutation(profiler_module):
    original = make_profiler_config(
        enable_profiling=True,
        profile_freq=20,
        profiler_warmup=2,
        profiler_active=3,
        profiler_repeat=2,
        profiler_skip_first=4,
        profiler_skip_first_wait=1,
        save_traces_folder="traces",
        enable_memory_snapshot=True,
        memory_snapshot_freq=7,
        save_memory_snapshot_folder="snapshots",
        memory_snapshot_max_entries=1234,
    )
    before = {item.name: getattr(original, item.name) for item in fields(original)}
    config = profiler_module.npu_profiler(original)
    assert type(config) is profiler_module.NPUProfiler.Config
    assert type(config.build()) is profiler_module.NPUProfiler
    assert config is not original
    for name, value in before.items():
        assert getattr(config, name) == value
        assert getattr(original, name) == value


@pytest.mark.parametrize("original", [False, True])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_enable_profiling_override(profiler_module, original, enabled):
    config = profiler_module.npu_profiler(
        make_profiler_config(enable_profiling=original), enable_profiling=enabled
    )
    assert config.enable_profiling is (original if enabled is None else enabled)


def test_rank_sequences_are_copied_and_defaults_are_independent(profiler_module):
    ranks = [0, 2]
    config = profiler_module.npu_profiler(make_profiler_config(), ranks=ranks)
    ranks.append(3)
    assert config.ranks == [0, 2]
    assert profiler_module.npu_profiler(make_profiler_config(), ranks=(1, 2)).ranks == [
        1,
        2,
    ]
    first = profiler_module.NPUProfiler.Config()
    first.ranks.append(1)
    assert profiler_module.NPUProfiler.Config().ranks == [0]
    assert profiler_module.npu_profiler(make_profiler_config()).ranks == [0]


def test_override_json_and_direct_recipe_entry_points(profiler_module):
    original = make_profiler_config(profile_freq=20, enable_memory_snapshot=True)
    root = ProfilerRoot.Config(profiler=original)
    options = dict(enable_profiling=True, ranks=[0, 2], level="level2", with_stack=True)
    imports = parse_cli_imports([TARGET + "=" + json.dumps(options)])
    replacements = apply_overrides(OverrideConfig(imports=imports), root)
    assert len(replacements) == 1
    assert type(root.profiler.build()) is profiler_module.NPUProfiler
    for name, value in options.items():
        assert getattr(root.profiler, name) == value
    assert root.profiler.profile_freq == 20
    assert root.profiler.enable_memory_snapshot
    assert not original.enable_profiling
    direct = profiler_module.npu_profiler(original, **options)
    direct_root = ProfilerRoot.Config(profiler=direct)
    assert apply_overrides(OverrideConfig(), direct_root) == []
    assert direct_root.profiler is direct
    # exact=True must not replace an already configured NPU profiler.
    assert apply_overrides(OverrideConfig(imports=[TARGET]), direct_root) == []
    assert direct_root.profiler is direct


@pytest.mark.parametrize("name", ["typo", "profile_freq", "enable_memory_snapshot"])
def test_unknown_factory_and_override_options_fail(profiler_module, name):
    with pytest.raises(TypeError, match=name):
        profiler_module.npu_profiler(make_profiler_config(), **{name: True})
    root = ProfilerRoot.Config()
    original = root.profiler
    with pytest.raises(TypeError, match=name):
        apply_overrides(OverrideConfig(imports=[(TARGET, {name: True})]), root)
    assert root.profiler is original


@pytest.mark.parametrize(
    "options, message",
    [
        ({"start_step": 1}, "provided together"),
        ({"end_step": 2}, "provided together"),
        ({"start_step": -1, "end_step": 2}, "0 <= start_step < end_step"),
        ({"start_step": 2, "end_step": 2}, "0 <= start_step < end_step"),
        ({"start_step": 3, "end_step": 2}, "0 <= start_step < end_step"),
        ({"level": "level3"}, "profiler level"),
        ({"aic_metrics_type": "unknown"}, "aic_metrics_type"),
        ({"ranks": [-2]}, "ranks"),
        ({"ranks": [-1, 0]}, "ranks"),
    ],
)
def test_invalid_npu_configuration(profiler_module, options, message):
    with pytest.raises(ValueError, match=message):
        profiler_module.npu_profiler(make_profiler_config(), **options)


def test_periodic_schedule_validation_is_inherited(profiler_module):
    original = make_profiler_config(profile_freq=3)
    with pytest.raises(ValueError, match="profile_freq"):
        profiler_module.npu_profiler(original, enable_profiling=True)
    # An explicit window supplies its own schedule independently of profile_freq.
    config = profiler_module.npu_profiler(
        original, enable_profiling=True, start_step=0, end_step=1
    )
    assert config.profile_freq == 3


def test_cuspy_is_rejected(profiler_module):
    with pytest.raises(ValueError, match="cuspy"):
        profiler_module.npu_profiler(make_profiler_config(cuspy=CuspyProfilerConfig()))


@pytest.mark.parametrize("level", ["level0", "level1", "level2"])
def test_stack_requires_cpu(profiler_module, level):
    with pytest.raises(ValueError, match="with_stack=True requires with_cpu=True"):
        profiler_module.npu_profiler(
            make_profiler_config(), with_cpu=False, with_stack=True, level=level
        )


@pytest.mark.parametrize("level", ["level1", "level2"])
def test_shapes_require_cpu_at_higher_levels(profiler_module, level):
    with pytest.raises(ValueError, match="record_shapes=True requires with_cpu=True"):
        profiler_module.npu_profiler(
            make_profiler_config(), with_cpu=False, record_shapes=True, level=level
        )


@pytest.mark.parametrize("level", ["level0", "level1", "level2"])
def test_cpu_allows_shapes_and_stack(profiler_module, level):
    config = profiler_module.npu_profiler(
        make_profiler_config(),
        with_cpu=True,
        record_shapes=True,
        with_stack=True,
        level=level,
    )
    assert config.record_shapes and config.with_stack


def test_level0_allows_npu_only_shapes(profiler_module):
    config = profiler_module.npu_profiler(
        make_profiler_config(), with_cpu=False, record_shapes=True, level="level0"
    )
    assert config.record_shapes and not config.with_cpu


def test_default_backend_options(runtime):
    prof = runtime.build()
    assert prof.kwargs["activities"] == ["NPU", "CPU"]
    for name in ("record_shapes", "with_stack", "profile_memory"):
        assert prof.kwargs[name] is False
    assert prof.kwargs["experimental_config"] == dict(
        profiler_level="level1",
        aic_metrics="PipeUtilization",
        data_simplification=False,
    )
    assert (
        prof.kwargs["on_trace_ready"]
        is runtime.npu.tensorboard_trace_handler.return_value
    )
    assert prof.events == [("enter", 0, ProfilerAction.NONE)]


@pytest.mark.parametrize("level", ["level0", "level1", "level2"])
def test_cpu_shapes_stack_and_explicit_backend_options(runtime, level):
    prof = runtime.build(
        level=level,
        aic_metrics_type="ArithmeticUtilization",
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        data_simplification=True,
        analyse_flag=False,
    )
    assert prof.kwargs["activities"] == ["NPU", "CPU"]
    for name in ("record_shapes", "with_stack", "profile_memory"):
        assert prof.kwargs[name] is True
    assert prof.kwargs["experimental_config"] == dict(
        profiler_level=level,
        aic_metrics="ArithmeticUtilization",
        data_simplification=True,
    )
    runtime.npu.tensorboard_trace_handler.assert_called_once_with(
        os.path.join("base", "test-traces"),
        worker_name="rank0",
        analyse_flag=False,
    )


def test_npu_only_shape_and_memory_options(runtime):
    prof = runtime.build(
        with_cpu=False, record_shapes=True, level="level0", profile_memory=True
    )
    assert prof.kwargs["activities"] == ["NPU"]
    assert prof.kwargs["record_shapes"]
    assert prof.kwargs["profile_memory"]


@pytest.mark.parametrize("enabled, ranks", [(False, [0]), (True, [1]), (True, [])])
def test_disabled_or_unselected_does_not_create_backend(
    profiler_module, runtime, enabled, ranks
):
    owner = profiler_module.npu_profiler(
        make_profiler_config(), enable_profiling=enabled, ranks=ranks
    ).build()
    assert (
        owner.build_torch_profiler(global_step=0, base_folder="", leaf_folder="")
        is None
    )
    runtime.npu.schedule.assert_not_called()
    runtime.npu.tensorboard_trace_handler.assert_not_called()
    runtime.npu.profile.assert_not_called()


@pytest.mark.parametrize("ranks", [None, [-1], [2], [0, 2]])
def test_selected_and_all_ranks(runtime, ranks):
    runtime.rank.return_value = 2
    assert isinstance(runtime.build(ranks=ranks), FakeProfile)
    assert (
        runtime.npu.tensorboard_trace_handler.call_args.kwargs["worker_name"] == "rank2"
    )


def test_uninitialized_process_group_uses_rank_zero(runtime):
    runtime.initialized.return_value = False
    assert isinstance(runtime.build(), FakeProfile)
    runtime.rank.assert_not_called()
    assert (
        runtime.npu.tensorboard_trace_handler.call_args.kwargs["worker_name"] == "rank0"
    )


def test_periodic_schedule_records_every_tenth_training_step(runtime):
    prof = runtime.build()
    runtime.npu.schedule.assert_called_once_with(wait=6, warmup=3, active=1)
    actions = training_actions(prof, 21)
    assert [action for _, action in actions[:10]] == (
        [ProfilerAction.NONE] * 6
        + [ProfilerAction.WARMUP] * 3
        + [ProfilerAction.RECORD_AND_SAVE]
    )
    assert recorded_steps(actions) == [10, 20]


@pytest.mark.parametrize(
    "skip_wait, expected", [(0, [6, 7, 11, 12]), (1, [4, 5, 9, 10])]
)
def test_custom_schedule_repeat_and_skip(runtime, skip_wait, expected):
    original = make_profiler_config(
        profile_freq=5,
        profiler_warmup=1,
        profiler_active=2,
        profiler_repeat=2,
        profiler_skip_first=2,
        profiler_skip_first_wait=skip_wait,
    )
    prof = runtime.build(original=original)
    runtime.npu.schedule.assert_called_once_with(
        wait=2, warmup=1, active=2, repeat=2, skip_first=2, skip_first_wait=skip_wait
    )
    assert recorded_steps(training_actions(prof, 20)) == expected


def test_unlimited_repeat_and_nonzero_skip_first_wait(runtime):
    original = make_profiler_config(
        profile_freq=5,
        profiler_warmup=1,
        profiler_active=2,
        profiler_repeat=0,
        profiler_skip_first_wait=2,
    )
    prof = runtime.build(original=original)
    assert recorded_steps(training_actions(prof, 13)) == [2, 3, 7, 8, 12, 13]


@pytest.mark.parametrize("restored", [5, 9, 10, 19])
def test_resume_aligns_initial_action_and_future_cycles(runtime, restored):
    prof = runtime.build(global_step=restored)
    assert prof.step_num == restored
    assert prof.events[0] == ("enter", restored, prof.current_action)
    expected_initial = {
        5: ProfilerAction.NONE,
        9: ProfilerAction.RECORD_AND_SAVE,
        10: ProfilerAction.NONE,
        19: ProfilerAction.RECORD_AND_SAVE,
    }
    assert prof.current_action is expected_initial[restored]
    assert recorded_steps(training_actions(prof, 21, restored_step=restored)) == [
        step for step in range(restored + 1, restored + 22) if step % 10 == 0
    ]


def test_resume_inside_multistep_recording_window(runtime):
    original = make_profiler_config(
        profile_freq=5, profiler_warmup=1, profiler_active=3
    )
    prof = runtime.build(original=original, global_step=3)
    assert prof.events[0] == ("enter", 3, ProfilerAction.RECORD)
    assert recorded_steps(training_actions(prof, 7, restored_step=3)) == [
        4,
        5,
        8,
        9,
        10,
    ]


def test_resume_after_finite_repeat_does_not_restart(runtime):
    prof = runtime.build(
        original=make_profiler_config(profiler_repeat=1), global_step=20
    )
    assert prof.current_action is ProfilerAction.NONE
    assert recorded_steps(training_actions(prof, 20, restored_step=20)) == []


@pytest.mark.parametrize("restored", [0, 8, 9, 10, 1000])
def test_absolute_single_recording_schedule_across_resume(runtime, restored):
    original = make_profiler_config(
        profile_freq=1,
        profiler_warmup=0,
        profiler_active=1,
        profiler_repeat=1,
        profiler_skip_first=8,
    )
    prof = runtime.build(original=original, global_step=restored)
    assert recorded_steps(training_actions(prof, 12, restored_step=restored)) == (
        [9] if restored < 9 else []
    )


@pytest.mark.parametrize("restored", [0, 9, 100])
@pytest.mark.parametrize("start, end", [(0, 1), (5, 6), (2, 5)])
def test_relative_window_restarts_from_run_start(runtime, restored, start, end):
    original = make_profiler_config(
        profiler_repeat=3, profiler_skip_first=20, enable_memory_snapshot=True
    )
    prof = runtime.build(
        original=original, global_step=restored, start_step=start, end_step=end
    )
    runtime.npu.schedule.assert_called_once_with(
        wait=0, warmup=0, active=end - start, repeat=1, skip_first=start
    )
    assert prof.step_num == 0
    assert recorded_steps(training_actions(prof, 15, restored_step=restored)) == list(
        range(restored + start + 1, restored + end + 1)
    )
    assert original.profiler_repeat == 3
    assert original.profiler_skip_first == 20
    assert original.enable_memory_snapshot


@pytest.mark.parametrize("explicit", [None, "relative-traces", "absolute"])
@pytest.mark.parametrize("save_folder", ["custom-traces", "absolute"])
def test_output_paths_and_explicit_save_dir(runtime, explicit, save_folder):
    if save_folder == "absolute":
        save_folder = os.path.abspath("public-traces")
    if explicit == "absolute":
        explicit = os.path.abspath("npu-traces")
    runtime.build(
        original=make_profiler_config(save_traces_folder=save_folder),
        save_dir=explicit,
        leaf_folder="replica",
    )
    expected = os.path.join("base", save_folder) if explicit is None else explicit
    runtime.npu.tensorboard_trace_handler.assert_called_once_with(
        expected, worker_name="rank0", analyse_flag=True
    )


def test_inherited_lifecycle_steps_and_cleans_up_on_error(profiler_module, runtime):
    memory = Mock()
    owner = profiler_module.npu_profiler(
        make_profiler_config(enable_memory_snapshot=True), enable_profiling=True
    ).build()
    error = RuntimeError("training failed")
    with patch.object(
        Profiler, "build_memory_profiler", return_value=memory
    ) as builder:
        assert (
            owner.active(global_step=9, base_folder="run", leaf_folder="replica")
            is owner
        )
        with pytest.raises(RuntimeError, match="training failed"):
            with owner as entered:
                assert entered is owner
                backend = owner.torch_profiler
                owner.step()
                raise error
        builder.assert_called_once_with(
            global_step=9, base_folder="run", leaf_folder="replica"
        )
    assert backend.events[:2] == [
        ("enter", 9, ProfilerAction.RECORD_AND_SAVE),
        ("step", 9, ProfilerAction.RECORD_AND_SAVE),
    ]
    assert backend.events[-1][:3] == ("exit", RuntimeError, error)
    assert backend.events[-1][3] is not None
    memory.step.assert_called_once_with()
    assert owner.torch_profiler is None
    assert owner.memory_profiler is None


@pytest.mark.parametrize("enabled, ranks", [(False, [0]), (True, [1])])
def test_memory_builder_is_independent_of_trace_selection(
    profiler_module, runtime, enabled, ranks
):
    owner = profiler_module.npu_profiler(
        make_profiler_config(enable_memory_snapshot=True, memory_snapshot_freq=7),
        enable_profiling=enabled,
        ranks=ranks,
        save_dir="only-traces",
    ).build()
    memory = Mock()
    with patch.object(
        Profiler, "build_memory_profiler", return_value=memory
    ) as builder:
        with owner.active(global_step=9, base_folder="base", leaf_folder="leaf"):
            assert owner.torch_profiler is None
            assert owner.memory_profiler is memory
            owner.step()
        builder.assert_called_once_with(
            global_step=9, base_folder="base", leaf_folder="leaf"
        )
    memory.step.assert_called_once_with()
    runtime.npu.profile.assert_not_called()
    assert owner.memory_profiler is None
