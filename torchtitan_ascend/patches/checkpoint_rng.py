# SPDX-License-Identifier: Apache-2.0
"""Initialize host generators and save each rank's RNG in native DCP checkpoints.

The rank-keyed format preserves different random streams across data, tensor,
context and pipeline ranks. Restoring it requires the same rank topology.
Model-only and Hugging Face exports retain the upstream save policy.
"""

import pickle
import random
from functools import wraps

from torchtitan_ascend.patches import patch


@patch(
    target="torchtitan.training_engine:TrainingEngine.__init__",
    kind="wrap",
    deferred=True,
)
def initialize_host_rng(original):
    """Seed host generators after native mesh-aware setup, before model creation."""

    @wraps(original)
    def initialize(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self.device.type in {"cpu", "npu"} and self.config.debug.seed is not None:
            import numpy as np
            import torch

            # The native setup has already applied distinct mesh-axis offsets.
            seed = torch.initial_seed()
            random.seed(seed)
            np.random.seed(seed % 2**32)
        return result

    return initialize


class RankRNGState:
    """DCP state for Python, NumPy, Torch CPU and the worker's accelerator."""

    def __init__(self, device):
        import torch.distributed as dist

        self.device = device
        self.rank_key = f"rank_{dist.get_rank()}"

    def state_dict(self):
        """Capture generators at the same save boundary as other trainer state."""
        import numpy as np
        import torch

        row = {
            "python": pickle.dumps(random.getstate()),
            "numpy": pickle.dumps(np.random.get_state()),
            "cpu": torch.get_rng_state(),
        }
        if self.device.type != "cpu":
            row["device"] = torch.get_device_module(self.device.type).get_rng_state(
                self.device
            )
        return {self.rank_key: row}

    def load_state_dict(self, state):
        """Restore this rank's stream from a trusted native training checkpoint."""
        import numpy as np
        import torch

        row = state[self.rank_key]
        # These bytes are the native RNG objects saved above, not model tensors.
        random.setstate(pickle.loads(row["python"]))  # nosec B301
        np.random.set_state(pickle.loads(row["numpy"]))  # nosec B301
        torch.set_rng_state(row["cpu"])
        if self.device.type != "cpu":
            torch.get_device_module(self.device.type).set_rng_state(
                row["device"], self.device
            )


@patch(
    target="torchtitan.components.checkpointer.dcp:CheckpointManager.__init__",
    kind="wrap",
    deferred=True,
)
def apply(original):
    """Register RNG after worker device binding and before checkpoint loading."""

    @wraps(original)
    def initialize(self, *args, **kwargs):
        import torch

        accelerator = torch.accelerator.current_accelerator(check_available=True)
        if accelerator is not None and accelerator.type == "npu":
            # FSDP CPU offload changes parameter storage, not the worker device.
            device = torch.device("npu", torch.accelerator.current_device_index())
        else:
            device = next(
                parameter.device
                for part in kwargs["model_parts"]
                for parameter in part.parameters()
            )
        if device.type in {"cpu", "npu"}:
            extra_states = dict(kwargs["extra_states"])
            if "rng" in extra_states:
                raise ValueError("Checkpoint extra state rng is already registered")
            extra_states["rng"] = RankRNGState(device)
            kwargs["extra_states"] = extra_states
        return original(self, *args, **kwargs)

    return initialize
