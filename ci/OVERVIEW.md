# TorchTitan-Ascend CI Docker Overview

[English](OVERVIEW.md) | [简体中文](OVERVIEW.zh.md)

## Image Information

| Target | Image Name | Base Image |
| ------ | ------ | ------ |
| CPU | `torchtitan-ascend-ci:cpu` | `python:3.12-slim-bookworm` |
| A2 NPU | `torchtitan-ascend-ci:a2` | `ascendai/cann:9.2.0-910b-ubuntu22.04-py3.12` |

The build entrypoint is `ci/build.sh`, the Dockerfile is `ci/Dockerfile`, and the default architecture is `linux/arm64`. The images install TorchTitan at `/opt/torchtitan` and this repository at `/opt/torchtitan-ascend`. Tests run from `/opt/torchtitan`.

By default, the image clones `refactor/npu-override` from the [official repository](https://gitcode.com/Ascend/torchtitan-ascend). Containers use the source and test lists included in the image.

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

Docker build caching is enabled by default. To fetch the branch again, append `--no-cache`, for example `bash ci/build.sh cpu --no-cache` (also applicable to A2).

## Run Tests

Run these commands from any directory on the host to start a container and execute the corresponding test list included in the image:

| Target | Test List |
| ------ | ------ |
| CPU | `ci/testlists/cpu.txt` |
| A2 | `ci/testlists/a2.txt` |

The image's default entrypoint automatically configures the environment and copies this repository's adapted tests from the image into TorchTitan's `tests/` directory: CPU copies `tests/unit_tests/cpu/`, and A2 copies `tests/`. There is no need to run `entrypoint.sh` manually.

### CPU

```bash
docker run --rm torchtitan-ascend-ci:cpu \
    bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu
```

### A2

The complete A2 list requires eight visible NPUs. Run the following command in Bash on an A2 host with the Ascend driver installed and device IDs `0..7`:

```bash
docker run --rm --network host --ipc host \
    --device=/dev/davinci{0..7} \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    torchtitan-ascend-ci:a2 \
    bash /opt/torchtitan-ascend/ci/run-testlist.sh a2
```

Inside a container started through the image's default entrypoint, run the script for its target directly:

```bash
# Inside the CPU container
bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu

# Inside the A2 container
bash /opt/torchtitan-ascend/ci/run-testlist.sh a2
```

Distributed tests start their own workers. Do not wrap the entire list in another `torchrun` invocation.

## Script Arguments and Results

```bash
# Collect CPU tests inside the container, with 100 entries per batch
BATCH=100 bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu --collect-only
```

| Argument | Meaning | Default |
| ------ | ------ | ------ |
| `cpu` / `a2` | Run the corresponding list | `cpu` |
| `BATCH` | Entries per batch; must be a positive integer | `150` |
| `WORKDIR` | TorchTitan test working directory | `/opt/torchtitan` |
| pytest arguments | Passed to each pytest invocation, such as `--collect-only` | None |

The script skips blank lines and comments. If a batch fails, every entry in that batch runs again individually. Entries that still fail are summarized as `FAILED_OR_ERROR`.

Exit codes: `0` means no final failures were reported, `1` means failures remain after retries, and `2` means invalid arguments or test lists. Preserve the pytest logs in CI and use pytest's `skipped` summary to identify skipped tests.

## Add Tests

### 1. Prepare Test Files and Lists on a Personal Branch

| Target | Test File Location in This Repository | Test List |
| ------ | ------ | ------ |
| CPU | Matching path under `tests/unit_tests/cpu/` | `ci/testlists/cpu.txt` |
| A2 | Same relative path as upstream under `tests/`, such as `tests/unit_tests/gpu/` | `ci/testlists/a2.txt` |

Make changes on a personal branch in the host checkout. Prepare the complete test file when adaptation is required. When reusing an unchanged test from the pinned TorchTitan version, add only its list entry. CPU tests must run without NPU packages installed.

Add the node ID to the corresponding TXT file's test-file group. Add a group comment for a new test file. Write one node ID per line, for example:

```text
tests/unit_tests/cpu/config/test_configurable.py::TestConfigurable::test_valid_config
```

This entry already exists and only demonstrates the format. Check for duplicates before adding an entry. Do not include `pytest`, command-line arguments, quotes, or inline comments in the list. New dependencies also require installation configuration changes.

### 2. Commit, Push, and Build the Personal Branch Image

From the repository root on the host, review the test and list changes, then commit and push them to the personal branch:

```bash
git add tests/ ci/testlists/
# If installation or dependency configuration changed, git add those files too
git commit -m "test: add CPU or A2 CI cases"
git push origin HEAD
```

Replace the HTTPS URL below with the clone URL of the repository containing that branch, and replace the branch name. The build machine and container must be able to access the URL. The selected branch must include the CI scripts and test lists.

```bash
# CPU; change the final cpu to a2 when validating A2
ASCEND_REPO='https://gitcode.com/YOUR_ACCOUNT/torchtitan-ascend.git' \
ASCEND_BRANCH='your-test-branch' \
    bash ci/build.sh cpu
```

### 3. Start a Validation Container

Run the command for the target from any directory on the host. The image's default entrypoint initializes the environment and copies the test files automatically.

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

### 4. Collect and Run the Test Individually

Run inside the container. This example uses an existing CPU test; replace it with your test file and node ID:

```bash
cd /opt/torchtitan
pytest tests/unit_tests/cpu/config/test_configurable.py --collect-only -q
pytest 'tests/unit_tests/cpu/config/test_configurable.py::TestConfigurable::test_valid_config' -vv -rs
```

Check the exact node IDs in the list against the collection output. To select one parameterized case, retain its complete `[parameter]` suffix and quote the node ID in shell commands. If it is skipped, first check the device, data, and dependency requirements.

Local edits and later remote commits do not update existing images automatically. If tests, lists, or dependencies need changes, return to step 1, commit, push, rebuild with `--no-cache`, and start a new container for validation.

### 5. Validate the Complete List and Submit a PR

Inside the corresponding container, run the complete list for its target:

```bash
# Inside the CPU container
bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu

# Inside the A2 container
bash /opt/torchtitan-ascend/ci/run-testlist.sh a2
```

After the individual test and complete list pass, submit a PR from the personal branch to the official target branch. For an existing PR, keep using its original branch and report the new validation results. Include the image's source commit, executed commands, and pytest pass, fail, and skip results in the PR. Skipped tests alone do not establish successful validation.

## License

This repository is released under Apache License 2.0. See [LICENSE](../LICENSE). Third-party licenses are provided in [LICENSE-BSD-3-Clause](../LICENSE-BSD-3-Clause) and [Third_Party_Open_Source_Software_Notice](../Third_Party_Open_Source_Software_Notice).
