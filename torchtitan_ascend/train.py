# SPDX-License-Identifier: Apache-2.0
"""Apply registered fixes, then delegate training and CLI parsing to TorchTitan."""


def main() -> None:
    from torchtitan_ascend import setup

    setup()
    # Upstream imports torch, which automatically loads the installed NPU backend.
    from torchtitan.train import main as titan_main

    titan_main()


if __name__ == "__main__":
    main()
