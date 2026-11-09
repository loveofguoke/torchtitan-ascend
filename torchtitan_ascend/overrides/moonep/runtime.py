# SPDX-License-Identifier: Apache-2.0
"""Resident expert homes using only MoonEP's public Ascend interfaces."""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import time
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from .settings import MoonEPConfig, validate_endpoint

logger = logging.getLogger(__name__)
_runtime: MoonEPRuntime | None = None
_retired = False


def _projection(full_projection, **kwargs):
    from moonep.buffer import projection_home_slice, projection_slot_slice

    home, slots = projection_home_slice(full_projection), projection_slot_slice(
        full_projection
    )
    num_elements_per_group = home.shape[1] * home.shape[2]
    group_indices = torch.cat(
        tuple(
            torch.arange(group_slice.shape[0], device=group_slice.device)
            + (group_slice.storage_offset() - full_projection.storage_offset())
            // num_elements_per_group
            for group_slice in (home, slots)
        )
    )
    return SimpleNamespace(
        full_projection=full_projection,
        home=home,
        slots=slots,
        group_indices=group_indices,
        **kwargs,
    )


def _fingerprint(state_value):
    return int.from_bytes(
        hashlib.sha256(state_value.encode()).digest()[:8], "little"
    ) % (2**63)


def _bootstrap_gather(ep_group, device, state_values):
    local_state = torch.tensor(state_values, dtype=torch.int64, device=device)
    gathered_states = torch.empty(
        ep_group.size() * len(state_values), dtype=torch.int64, device=device
    )
    dist.all_gather_into_tensor(gathered_states, local_state, group=ep_group)
    return gathered_states.view(ep_group.size(), -1).cpu().tolist()


def _bootstrap_endpoint(ep_group, device, endpoint):
    # GROUP_RANK identifies a torchrun node even when containers share a hostname.
    node_id = os.environ.get("GROUP_RANK")
    node_id = (
        f"torchrun:{node_id}"
        if node_id is not None
        else f"hostname:{socket.gethostname()}"
    )
    endpoint_error = None
    try:
        endpoint = validate_endpoint(endpoint) if endpoint is not None else None
    except ValueError as validation_error:
        endpoint_error = validation_error
    rank_states = _bootstrap_gather(
        ep_group,
        device,
        [
            _fingerprint(node_id),
            _fingerprint(str(endpoint)),
            int(endpoint_error is not None),
        ],
    )
    if any(rank_state[2] for rank_state in rank_states):
        raise ValueError(
            "Invalid MoonEP shmem_ip_port on an EP rank"
        ) from endpoint_error
    if len({rank_state[1] for rank_state in rank_states}) != 1:
        raise ValueError(
            "MoonEP EP ranks must use the same shmem_ip_port or all use automatic bootstrap"
        )
    if endpoint is not None:
        return endpoint
    if len({rank_state[0] for rank_state in rank_states}) == 1:
        return None
    # All ranks take this second collective, including ranks with invalid env.
    try:
        master_addr, master_port = os.environ.get("MASTER_ADDR"), os.environ.get(
            "MASTER_PORT"
        )
        if not master_addr or not master_port:
            raise ValueError(
                "Multi-host MoonEP requires shmem_ip_port or MASTER_ADDR and MASTER_PORT"
            )
        host_literal = (
            f"[{master_addr}]"
            if ":" in master_addr and not master_addr.startswith("[")
            else master_addr
        )
        endpoint = validate_endpoint(f"tcp://{host_literal}:{int(master_port) + 1}")
    except ValueError as validation_error:
        endpoint_error = validation_error
    rank_states = _bootstrap_gather(
        ep_group, device, [_fingerprint(str(endpoint)), int(endpoint_error is not None)]
    )
    if any(rank_state[1] for rank_state in rank_states):
        raise ValueError(
            "Invalid MoonEP MASTER_ADDR/MASTER_PORT on an EP rank; set shmem_ip_port explicitly"
        ) from endpoint_error
    if len({rank_state[0] for rank_state in rank_states}) != 1:
        raise ValueError(
            "MoonEP EP ranks derived different bootstrap endpoints; set shmem_ip_port explicitly"
        )
    return endpoint


class MoonEPRuntime:
    """Own one process's heap; layers have independent weight slots and shared gradients."""

    def __init__(
        self,
        ep_group,
        num_max_tokens_per_rank,
        hidden_dim,
        top_k,
        num_experts,
        **options,
    ):
        self.resource_key = (
            ep_group,
            num_max_tokens_per_rank,
            hidden_dim,
            top_k,
            num_experts,
        )
        self.communication_options = MoonEPConfig(**options).options()
        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(ep_group)
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_group.size()
        self.weight_projections = []
        self.gradient_pools = []
        self.buffer: Any = None
        self.owns_heap = self.started = self.closed = self.incomplete = False

    def allocate(self, projection_shapes):
        """Register all communication and gradient resources before heap initialization."""
        if self.closed:
            raise RuntimeError("MoonEP runtime is closed")
        if self.owns_heap:
            if self.incomplete or self.projection_shapes != projection_shapes:
                raise ValueError(
                    "MoonEP allocation is incomplete or has different projection shapes"
                )
            return
        from moonep import Buffer, ShmemRuntime
        from moonep.buffer import create_full_grad_and_reduce_buffer

        if ShmemRuntime.is_initialized():
            raise RuntimeError("MoonEP requires ownership of the process SHMEM runtime")
        _, num_max_tokens_per_rank, hidden_dim, top_k, num_experts = self.resource_key
        self.owns_heap = self.incomplete = True
        self.projection_shapes = projection_shapes
        if self.communication_options["planning_use_udma"]:
            logger.warning(
                "MoonEP planning uses MTE; planning_use_udma does not enable UDMA planning"
            )
        self.trace("allocate.buffer.begin")
        self.buffer = self._create(
            "buffer",
            Buffer,
            S=num_max_tokens_per_rank,
            H=hidden_dim,
            K=top_k,
            E=num_experts,
            B=self.num_local_experts,
            num_ep_ranks=self.ep_group.size(),
            group=self.ep_group,
            explicitly_destroy=True,
            use_udma=self.communication_options["use_udma"],
            planning_use_udma=self.communication_options["planning_use_udma"],
            grad_reduce_use_udma=self.communication_options["grad_reduce_use_udma"],
            num_sms=self.communication_options["num_sms"],
            comm_stream_priority=self.communication_options["comm_stream_priority"],
        )
        self.trace("allocate.buffer.end")
        if self.communication_options["allocation_diagnostics"]:
            logger.warning(
                "MoonEP rank=%d buffer_symmetric_bytes=%d",
                self.ep_rank,
                self.buffer.symmetric_footprint,
            )
        for projection_shape in projection_shapes:
            # Builders copy local slices; expanded zero tables need just one scalar.
            zero_scalar = torch.zeros((), dtype=torch.float32, device="npu")
            self.trace("allocate.grad.begin", str(projection_shape))
            full_projection, reduce_carrier = self._create(
                "gradient",
                create_full_grad_and_reduce_buffer,
                num_experts,
                self.num_local_experts,
                *projection_shape,
                self.ep_rank,
                self.ep_group.size(),
                zero_scalar.expand(num_experts, *projection_shape),
                zero_scalar.expand(
                    self.ep_group.size(), self.num_local_experts, *projection_shape
                ),
            )
            self.gradient_pools.append(
                _projection(full_projection, reduce_carrier=reduce_carrier)
            )
            # Keep public carrier metadata while aliasing local [B,M,N] slots.
            reduce_carrier.set_(self.gradient_pools[-1].slots)
            self.describe("gradient", self.gradient_pools[-1])
        self.incomplete = False

    def bind(self, w13_weight, w2_weight, ep_mesh):
        """Copy initialized shards once and register original-shaped views of symmetric home."""
        if (
            self.started
            or self.closed
            or self.incomplete
            or len(self.gradient_pools) != 2
        ):
            raise RuntimeError("Bind all MoonEP parameters before starting SHMEM")
        from moonep.buffer import create_full_weight

        local_weights = tuple(
            weight.to_local() if isinstance(weight, DTensor) else weight
            for weight in (w13_weight, w2_weight)
        )
        expected_local_shapes = (
            (
                self.num_local_experts,
                2,
                self.projection_shapes[0][0] // 2,
                self.projection_shapes[0][1],
            ),
            (self.num_local_experts, *self.projection_shapes[1]),
        )
        if any(
            local_weight.dtype != torch.bfloat16
            or tuple(local_weight.shape) != projection_shape
            for local_weight, projection_shape in zip(
                local_weights, expected_local_shapes, strict=True
            )
        ):
            raise ValueError(
                "MoonEP requires local BF16 shards shaped [e,2,F,D] and [e,D,F]"
            )
        self.incomplete = True
        weight_projections, home_parameters = [], []
        for projection_index, local_weight in enumerate(local_weights):
            projection_shape = self.projection_shapes[projection_index]
            self.trace("allocate.weight.begin", str(projection_shape))
            full_projection = self._create(
                "weight",
                create_full_weight,
                self.num_experts,
                self.num_local_experts,
                *projection_shape,
                self.ep_rank,
                self.ep_group.size(),
                local_weight.new_zeros(()).expand(self.num_experts, *projection_shape),
            )
            projection = _projection(full_projection)
            self.weight_projections.append(projection)
            self.describe("weight", projection)
            projection.home.copy_(
                local_weight.detach().reshape(self.num_local_experts, *projection_shape)
            )
            home_dtensor = DTensor.from_local(
                projection.home.view(local_weight.shape),
                ep_mesh,
                (Shard(0),),
                run_check=False,
            )
            home_parameters.append(
                torch.nn.Parameter(
                    home_dtensor,
                    requires_grad=(w13_weight, w2_weight)[
                        projection_index
                    ].requires_grad,
                )
            )
            weight_projections.append(projection)
        self.incomplete = False
        return tuple(weight_projections), tuple(home_parameters)

    def start(self):
        """Freeze descriptors after every layer has initialized its buffers."""
        if not self.started:
            from moonep import ShmemRuntime

            if self.closed or self.incomplete or not self.weight_projections:
                raise RuntimeError(
                    "Initialize MoonEP expert pools before starting SHMEM"
                )
            if ShmemRuntime.is_initialized():
                raise RuntimeError(
                    "Another owner initialized SHMEM before MoonEP start"
                )
            enable_udma = (
                self.communication_options["use_udma"]
                or self.communication_options["planning_use_udma"]
            )
            endpoint = _bootstrap_endpoint(
                self.ep_group,
                self.weight_projections[0].home.device,
                self.communication_options["shmem_ip_port"],
            )
            self.trace("bootstrap.begin", str(endpoint or "uid"))
            self.incomplete = True
            ShmemRuntime.init_with_buffer(
                group=self.ep_group,
                ip_port=endpoint,
                enable_udma=enable_udma,
                qp_count=self.communication_options["qp_count"] if enable_udma else 0,
            )
            self.incomplete = False
            self.started = True
            self.trace("bootstrap.end")
            logger.info(
                "MoonEP active: EP=%d, S=%d, E=%d, %s",
                self.ep_group.size(),
                self.resource_key[1],
                self.num_experts,
                self.communication_options,
            )

    def trace(self, stage, detail=""):
        """Optional host progress markers; synchronized markers deliberately serialize work."""
        if self.communication_options.get("trace") or self.communication_options.get(
            "trace_sync"
        ):
            if self.communication_options.get("trace_sync"):
                torch.npu.synchronize()
            logger.warning(
                "MoonEP rank=%d t=%.6f %s %s",
                self.ep_rank,
                time.monotonic(),
                stage,
                detail,
            )

    def _create(self, role, builder, *args, **kwargs):
        try:
            return builder(*args, **kwargs)
        except Exception:
            if self.communication_options["allocation_diagnostics"]:
                logger.exception(
                    "MoonEP allocation failed: role=%s rank=%d EP=%d E=%d shapes=%s; resources retained for process exit",
                    role,
                    self.ep_rank,
                    self.ep_group.size(),
                    self.num_experts,
                    self.projection_shapes,
                )
            raise

    def describe(self, role, projection):
        """Report public tensor layouts, without probing or allocating private VMM resources."""
        if self.communication_options["allocation_diagnostics"]:
            logger.warning(
                "MoonEP rank=%d allocation=%s full=%s home=%s slots=%s physical_tensor_bytes=%d carrier_bytes=%d",
                self.ep_rank,
                role,
                tuple(projection.full_projection.shape),
                tuple(projection.home.shape),
                tuple(projection.slots.shape),
                sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (projection.home, projection.slots)
                ),
                (
                    projection.reduce_carrier.numel()
                    * projection.reduce_carrier.element_size()
                    if hasattr(projection, "reduce_carrier")
                    else 0
                ),
            )

    def call(self, method, *args, wait=True, **kwargs):
        """Order consumers after events, or return an event for a later projection consumer."""
        async_finish = self.communication_options["async_finish"]
        self.trace(f"{method}.submit")
        result = getattr(self.buffer, method)(
            *args, **kwargs, async_finish=async_finish
        )
        if async_finish and wait:
            event = result[-1] if isinstance(result, tuple) else result
            self.trace(f"{method}.wait")
            event.wait()
            if method == "dispatch":
                return result[:-1]
        self.trace(f"{method}.return")
        return result

    def _shmem_barrier(self):
        """Drain device work, then take the SHMEM barrier that also completes pending RMA for remote readers."""
        from moonep import ShmemRuntime

        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.ep_group)

    def prefetch(self, comm_plan, weight_projections):
        """Launch projections in consumption order; wait only when each GEMM needs its rows."""
        if self.communication_options["prefetch_sync"]:
            self.trace("prefetch.publish.begin")
            self._shmem_barrier()
            self.trace("prefetch.publish.end")
        if not self.communication_options["async_finish"]:
            self.call(
                "prefetch_projections",
                plan=comm_plan,
                projections=tuple(
                    projection.full_projection for projection in weight_projections
                ),
            )
            return (
                (projection.full_projection, projection.group_indices)
                for projection in weight_projections
            )
        prefetch_events = tuple(
            self.call(
                "prefetch_projections",
                plan=comm_plan,
                projections=(projection.full_projection,),
                wait=False,
            )
            for projection in weight_projections
        )

        def ready_projections():
            for projection_index, (projection, event) in enumerate(
                zip(weight_projections, prefetch_events, strict=True)
            ):
                self.trace("prefetch.consume", str(projection_index))
                event.wait()
                yield projection.full_projection, projection.group_indices

        return ready_projections()

    def reduce(self, comm_plan, grouped_weight_grads, weight_dtypes):
        """Publish FP32 slots and return independent, original-dtype home gradients."""
        for grad_pool, grouped_grad in zip(
            self.gradient_pools, grouped_weight_grads, strict=True
        ):
            grad_pool.home.copy_(grouped_grad[: self.num_local_experts])
            grad_pool.slots.copy_(grouped_grad[self.num_local_experts :])
        self.trace("grad.publish.begin")
        grad_publish = self.communication_options["grad_publish"]
        if grad_publish in ("shmem_host", "strong"):
            self._shmem_barrier()
        if grad_publish in ("collective", "strong"):
            dist.all_reduce(
                torch.zeros(1, device=grouped_weight_grads[0].device),
                group=self.ep_group,
            )
        self.trace("grad.publish.end")
        self.call(
            "reduce_projection_grads",
            plan=comm_plan,
            projection_grads=tuple(
                (grad_pool.full_projection, grad_pool.reduce_carrier)
                for grad_pool in self.gradient_pools
            ),
        )
        return tuple(
            grad_pool.home.to(dtype=weight_dtype, copy=True)
            for grad_pool, weight_dtype in zip(
                self.gradient_pools, weight_dtypes, strict=True
            )
        )

    def close(self):
        """Drain work and use public teardown; uncertain construction requires process exit."""
        if self.closed:
            return
        if not self.owns_heap:
            self.closed = True
            return
        from moonep import ShmemRuntime
        from moonep.buffer import release_owned

        self.trace("close.begin")
        torch.npu.synchronize()
        if self.started:
            ShmemRuntime.finalize()
            self.started = False
        elif ShmemRuntime.is_initialized():
            raise RuntimeError(
                "SHMEM has another owner or incomplete startup; retain backing and exit"
            )
        if self.incomplete:
            raise RuntimeError(
                "MoonEP construction left uncertain allocations; retain resources and exit"
            )
        # Include owned aliases; release_owned deduplicates registered backing handles.
        self.incomplete = True
        self.buffer.destroy()  # Public Buffer failures must not be swallowed by release_owned.
        release_owned(
            projections=tuple(
                projection.full_projection
                for projection in (*self.weight_projections, *self.gradient_pools)
            ),
            buffers=(self.buffer,),
        )
        if not self.buffer.destroyed:
            raise RuntimeError("MoonEP Buffer release failed; exit the process")
        self.incomplete = False
        self.gradient_pools.clear()
        self.weight_projections.clear()
        self.closed = True
        self.trace("close.end")


def get_runtime(
    ep_group, num_max_tokens_per_rank, hidden_dim, top_k, num_experts, **options
):
    """Share one process runtime; all layers must use identical communication settings."""
    global _runtime
    resource_key = (ep_group, num_max_tokens_per_rank, hidden_dim, top_k, num_experts)
    communication_options = MoonEPConfig(**options).options()
    if _retired:
        raise RuntimeError("Start a fresh process for a new MoonEP runtime")
    if _runtime is None:
        if ep_group.size() < 2 or num_experts % ep_group.size():
            raise ValueError(
                "MoonEP requires EP > 1 and an expert count divisible by EP"
            )
        _runtime = MoonEPRuntime(*resource_key, **communication_options)
    elif (
        _runtime.resource_key != resource_key
        or _runtime.communication_options != communication_options
        or _runtime.closed
    ):
        raise ValueError(
            "All MoonEP layers must share shapes, EP group and communication settings"
        )
    return _runtime


def current_runtime():
    """Return the initialized process runtime."""
    if _runtime is None:
        raise RuntimeError("Initialize MoonEP model buffers before dispatch")
    return _runtime


def close_runtime():
    """Release MoonEP before the process group; public teardown is not a re-init contract."""
    global _runtime, _retired
    if _runtime is not None:
        _runtime.close()
        from .ops import _plans

        _plans.clear()
        _runtime = None
        _retired = True
