# SPDX-License-Identifier: Apache-2.0
"""Shared configuration for the trainer, dispatcher and process runtime."""

from dataclasses import dataclass, fields
from urllib.parse import urlsplit


@dataclass(kw_only=True)
class MoonEPConfig:
    """Settings applied at trainer build time, before native resources are created."""

    use_udma: bool = False
    planning_use_udma: bool = False
    async_finish: bool = False
    qp_count: int = 2
    shmem_ip_port: str | None = None
    num_sms: int | None = None
    comm_stream_priority: int = -1
    tokens_per_rank: int | None = None
    recompute_dispatch: bool = False
    validate_routing: bool = False
    grad_publish: str = "strong"
    prefetch_sync: bool = True
    trace: bool = False
    trace_sync: bool = False
    allocation_diagnostics: bool = False

    def validate(self):
        """Reject malformed settings without importing device or MoonEP libraries."""
        for field_name in (
            "use_udma",
            "planning_use_udma",
            "async_finish",
            "recompute_dispatch",
            "validate_routing",
            "prefetch_sync",
            "trace",
            "trace_sync",
            "allocation_diagnostics",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"MoonEP {field_name} must be a boolean")
        if type(self.qp_count) is not int or self.qp_count < 0:
            raise ValueError("MoonEP qp_count must be a nonnegative integer")
        for field_name in ("num_sms", "tokens_per_rank"):
            field_value = getattr(self, field_name)
            if field_value is not None and (
                type(field_value) is not int or field_value <= 0
            ):
                raise ValueError(
                    f"MoonEP {field_name} must be a positive integer or None"
                )
        if type(self.comm_stream_priority) is not int:
            raise ValueError("MoonEP comm_stream_priority must be an integer")
        if self.grad_publish not in ("collective", "shmem_host", "strong"):
            raise ValueError(
                "MoonEP grad_publish supports collective, shmem_host or strong; "
                "native stream barrier is not public"
            )
        if self.shmem_ip_port is not None:
            validate_endpoint(self.shmem_ip_port)

    def options(self):
        """Copy only MoonEP fields from a trainer setting or dispatcher config."""
        self.validate()
        return {
            config_field.name: getattr(self, config_field.name)
            for config_field in fields(MoonEPConfig)
        }


def validate_endpoint(endpoint):
    """Validate TCP endpoints and normalize IPv6 to the public tcp6 scheme."""
    error_message = (
        "MoonEP shmem_ip_port must be tcp://<host>:<port> or "
        "tcp6://[<ipv6>]:<port> with port in 1..65535"
    )
    try:
        parsed_endpoint = urlsplit(endpoint)
        is_valid = (
            parsed_endpoint.scheme in ("tcp", "tcp6")
            and parsed_endpoint.hostname
            and parsed_endpoint.port is not None
            and parsed_endpoint.port > 0
            and not any(character.isspace() for character in endpoint)
            and not (
                parsed_endpoint.path
                or parsed_endpoint.query
                or parsed_endpoint.fragment
                or parsed_endpoint.username
                or parsed_endpoint.password
            )
        )
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(error_message) from error
    if not is_valid:
        raise ValueError(error_message)
    host_literal = (
        f"[{parsed_endpoint.hostname}]"
        if ":" in parsed_endpoint.hostname
        else parsed_endpoint.hostname
    )
    scheme = "tcp6" if ":" in parsed_endpoint.hostname else parsed_endpoint.scheme
    return f"{scheme}://{host_literal}:{parsed_endpoint.port}"
