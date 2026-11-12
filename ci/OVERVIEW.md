# TorchTitan-Ascend CI Docker Overview

[English](OVERVIEW.md) | [简体中文](OVERVIEW.zh.md)

## Image Information

| Target | Image Name | Base Image |
| ------ | ------ | ------ |
| CPU | `torchtitan-ascend-ci:cpu` | `python:3.12-slim-bookworm` |
| A2 NPU | `torchtitan-ascend-ci:a2` | `ascendai/cann:9.2.0-910b-ubuntu22.04-py3.12` |

The build entrypoint is `ci/build.sh`, the Dockerfile is `ci/Dockerfile`, and the default architecture is `linux/arm64`. The images install TorchTitan at `/opt/torchtitan` and preinstall this repository and its dependencies at `/opt/torchtitan-ascend` as the baseline environment. Tests run from `/opt/torchtitan`.

The image build prepares the baseline from `refactor/npu-override` in the [official repository](https://gitcode.com/Ascend/torchtitan-ascend). Reuse the image across runs; each test uses the source, scripts, and lists checked out by the pipeline.

| Component | Version |
| ------ | ------ |
| Python | 3.12 |
| PyTorch / torchvision | `2.15.0+cpu` / `0.29.2+cpu` |
| TorchTitan | v0.4.0, commit `45e33cd413b64cb37a7840cfd3c9765d3c639ae1` |
| pytest | `8.3.5` |
| A2 dependencies | CANN `9.2.0`, torch_npu, Triton-Ascend, `fla_npu` (`ascend910b`) |

See [install.sh](install.sh) for the fixed commits and installation steps for A2 dependencies.

## Build Images

Run from the repository root on an aarch64 build machine with Docker installed and access to the remote source repository and dependency download sources.

| Build Variable | Purpose | Default |
| ------ | ------ | ------ |
| `ASCEND_REPO` | Remote HTTPS clone URL for this repository | `https://gitcode.com/Ascend/torchtitan-ascend.git` |
| `ASCEND_BRANCH` | Branch to clone | `refactor/npu-override` |
| `MAX_JOBS` | Source compilation parallelism; adjust to available memory | `32` |
| `CORPORATE_CA_FILE` | Path to a PEM CA certificate for the corporate HTTPS gateway | Auto-detect a host certificate |

```bash
# Build the CPU test image: torchtitan-ascend-ci:cpu
bash ci/build.sh cpu

# Build the A2 NPU test image: torchtitan-ascend-ci:a2
bash ci/build.sh a2
```

Routine code and test changes do not require rebuilding the image. Update the image when dependencies or compilation tools change; append `--no-cache` when a complete rebuild is needed.

## Run Tests

Run the following commands inside the container using the source at `/opt/torchtitan-ascend`. The pipeline should run `ci/run-testlist.sh` from its checkout for the current test run.

| Target | Test List |
| ------ | ------ |
| CPU | `ci/testlists/cpu.txt` |
| A2 single card | `ci/testlists/a2-1card.txt` |
| A2 two cards | `ci/testlists/a2-2card.txt` |
| A2 eight cards | `ci/testlists/a2-8card.txt` |

```bash
# CPU
bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu

# A2 single card
bash /opt/torchtitan-ascend/ci/run-testlist.sh a2-1card

# A2 two cards
bash /opt/torchtitan-ascend/ci/run-testlist.sh a2-2card

# A2 eight cards
bash /opt/torchtitan-ascend/ci/run-testlist.sh a2-8card
```

The script installs code from its repository, copies adapted tests, and runs the corresponding list. The A2 test runner loads CANN and the other runtime environments automatically, including when invoked through `docker exec`; no manual entrypoint invocation is required.

### Local CPU Validation

Run on the host, using the image's source at `/opt/torchtitan-ascend`:

```bash
docker run --rm torchtitan-ascend-ci:cpu \
    bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu
```

### Local A2 Validation

The following example runs the 8-card list. Run in Bash on a host with the Ascend driver installed and device IDs `0..7`:

```bash
docker run --rm --network host --ipc host \
    --device=/dev/davinci{0..7} \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    torchtitan-ascend-ci:a2 \
    bash /opt/torchtitan-ascend/ci/run-testlist.sh a2-8card
```

Distributed tests start their own workers. Do not wrap the entire list in another `torchrun` invocation.

## Script Arguments and Results

```bash
# From the checkout root inside the container: install, copy, and collect CPU tests
BATCH=100 bash ci/run-testlist.sh cpu --collect-only
```

| Argument | Meaning | Default |
| ------ | ------ | ------ |
| `cpu` / `a2-1card` / `a2-2card` / `a2-8card` | Run the corresponding list; allocate devices for the group | `cpu` |
| `BATCH` | Entries per batch; must be a positive integer | `150` |
| `WORKDIR` | TorchTitan test working directory | `/opt/torchtitan` |
| pytest arguments | Passed to each pytest invocation, such as `--collect-only` | None |

The list runner skips blank lines and comments. If a batch fails, every entry in that batch runs again individually. Entries that still fail are summarized as `FAILED_OR_ERROR`.

Exit codes: `0` means no final failures were reported, `1` means failures remain after retries, and `2` means invalid arguments or test lists. Preserve the pytest logs in CI and use pytest's `skipped` summary to identify skipped tests.

## Add Upstream Tests

### 1. Prepare Test Files and Lists on a Personal Branch

| Target | Test File Location in This Repository | Test List |
| ------ | ------ | ------ |
| CPU | Matching path under `tests/unit_tests/cpu/` | `ci/testlists/cpu.txt` |
| A2 | Same relative path as upstream under `tests/`, such as `tests/unit_tests/gpu/` | `ci/testlists/a2-{1,2,8}card.txt` |

Make changes on a personal branch in the host checkout. Prepare the complete test file when adaptation is required. When reusing an unchanged test from the pinned TorchTitan version, add only its list entry. CPU tests must run without NPU packages installed. Verify that each test file and node ID exists in the pinned upstream version. Same-path adaptations are allowed; do not add repository-specific test cases.

Add the node ID to the corresponding TXT file's test-file group. Add a group comment for a new test file. Write one node ID per line, for example:

```text
tests/unit_tests/cpu/config/test_configurable.py::TestConfigurable::test_valid_config
```

This entry already exists and only demonstrates the format. Check for duplicates before adding an entry. Do not include `pytest`, command-line arguments, quotes, or inline comments in the list. New dependencies also require installation configuration changes.

Assign each new NPU test to `a2-1card.txt`, `a2-2card.txt`, or `a2-8card.txt` according to its actual device requirement (`world_size` for distributed tests). Register each entry in exactly one group to avoid duplicates.

### 2. Prepare the Target Test Image

Use the target CPU or A2 image; update the image when dependencies change.

### 3. Start a Validation Container

Run on the host. After entering the container, update `/opt/torchtitan-ascend` to the pushed commit under test before proceeding.

```bash
# CPU
docker run --rm -it torchtitan-ascend-ci:cpu bash
```

```bash
# A2
docker run --rm -it --network host --ipc host \
    --device=/dev/davinci{0..7} \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    torchtitan-ascend-ci:a2 bash
```

### 4. Collect and Check New Tests

Run inside the container. For A2 tests, replace `cpu` with the corresponding A2 list target:

```bash
cd /opt/torchtitan-ascend
bash ci/run-testlist.sh cpu --collect-only
```

Verify that the new node IDs are collected. Keep the complete `[parameter]` suffix for parameterized cases, without shell quotes in TXT lists. Adding a test file without registering its node IDs does not run it automatically. If a test is skipped, check its device, data, and dependency requirements.

### 5. Validate the Complete List and Submit a PR

Inside the corresponding container, run the complete list for its target:

```bash
# From the checkout root inside the CPU container
bash ci/run-testlist.sh cpu

# From the checkout root inside the A2 container, select the allocated device count
bash ci/run-testlist.sh a2-1card
bash ci/run-testlist.sh a2-2card
bash ci/run-testlist.sh a2-8card
```

After the complete list passes, commit the test files, matching list, and required code or dependency configuration, then submit a PR to the official target branch. Keep the original branch for an existing PR. Report the tested commit, image name, commands, and pytest pass, fail, and skip results in the PR. Skipped tests alone do not establish successful validation.

## License

This repository is released under Apache License 2.0. See [LICENSE](../LICENSE). Third-party licenses are provided in [LICENSE-BSD-3-Clause](../LICENSE-BSD-3-Clause) and [Third_Party_Open_Source_Software_Notice](../Third_Party_Open_Source_Software_Notice).
