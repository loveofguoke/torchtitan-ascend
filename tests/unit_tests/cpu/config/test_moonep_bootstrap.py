# SPDX-License-Identifier: Apache-2.0
"""Keep automatic SHMEM bootstrap scoped to each EP subgroup."""

import socket
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from torchtitan_ascend import setup

setup()

from torchtitan_ascend.overrides.moonep import runtime

MOCK_MASTER_ADDR = "10.42.0.25"
MOCK_EP_LEADER_ADDR = "10.42.0.26"


@pytest.mark.parametrize("leader_rank", [0, 2, 4])
@pytest.mark.parametrize("ep_rank", [0, 1])
def test_cross_host_groups_use_distinct_ports_and_their_own_leader(
    leader_rank, ep_rank
):
    group = SimpleNamespace(size=lambda: 2)
    leader = [0, 0, *socket.inet_pton(socket.AF_INET, MOCK_EP_LEADER_ADDR), *([0] * 12)]

    def gather(group_arg, device, values):
        assert group_arg is group and device == "cpu"
        if len(values) == 3:
            return [[1, *values[1:]], [2, *values[1:]]]
        if len(values) == 18:
            return [leader, values]
        return [values, values]

    route = MagicMock()
    route.__enter__.return_value.getsockname.return_value = (MOCK_EP_LEADER_ADDR, 10000)
    with (
        patch.dict("os.environ", MASTER_ADDR=MOCK_MASTER_ADDR, MASTER_PORT="29800"),
        patch.object(runtime, "_bootstrap_gather", side_effect=gather),
        patch.object(runtime.dist, "get_global_rank", return_value=leader_rank),
        patch.object(runtime.dist, "get_rank", return_value=ep_rank),
        patch.object(
            runtime.socket,
            "getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_DGRAM, 0, "", (MOCK_MASTER_ADDR, 29800))
            ],
        ),
        patch.object(runtime.socket, "socket", return_value=route) as create_socket,
    ):
        endpoint = runtime._bootstrap_endpoint(group, "cpu", None)
    assert endpoint == f"tcp://{MOCK_EP_LEADER_ADDR}:{29801 + leader_rank}"
    assert create_socket.call_count == int(ep_rank == 0)


@pytest.mark.parametrize("endpoint", [None, f"tcp://{MOCK_MASTER_ADDR}:30000"])
def test_single_host_keeps_native_uid_or_explicit_endpoint(endpoint):
    with patch.object(
        runtime,
        "_bootstrap_gather",
        side_effect=lambda group, device, values: [values, values],
    ):
        assert runtime._bootstrap_endpoint(None, "cpu", endpoint) == endpoint


def test_port_exhaustion_fails_collectively_before_native_init():
    def gather(group, device, values):
        if len(values) == 3:
            return [[1, *values[1:]], [2, *values[1:]]]
        return [values, values]

    with (
        patch.dict("os.environ", MASTER_ADDR=MOCK_MASTER_ADDR, MASTER_PORT="65534"),
        patch.object(runtime, "_bootstrap_gather", side_effect=gather),
        patch.object(runtime.dist, "get_global_rank", return_value=2),
    ):
        with pytest.raises(ValueError, match="Invalid MoonEP MASTER_ADDR/MASTER_PORT"):
            runtime._bootstrap_endpoint(None, "cpu", None)
