#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
    echo "Usage: $0 <cpu|a2> [image:tag] [docker build options...]"
    echo 'Environment: PLATFORM (default linux/arm64), MAX_JOBS (default 32),'
    echo '             CORPORATE_CA_FILE (PEM to trust inside the image)'
    echo '             ASCEND_REPO (default https://gitcode.com/Ascend/torchtitan-ascend.git),'
    echo '             ASCEND_BRANCH (default refactor/npu-override)'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi
target="$1"
shift
case "$target" in
    cpu|a2) ;;
    *) usage >&2; exit 2 ;;
esac
image="torchtitan-ascend-ci:$target"
if [[ $# -gt 0 && "$1" != -* ]]; then
    image="$1"
    shift
fi
platform="${PLATFORM:-linux/arm64}"
if [[ "$target" == a2 && "$platform" != linux/arm64 ]]; then
    echo 'The A2 target requires PLATFORM=linux/arm64.' >&2
    exit 2
fi

ascend_repo="${ASCEND_REPO:-https://gitcode.com/Ascend/torchtitan-ascend.git}"
ascend_branch="${ASCEND_BRANCH:-refactor/npu-override}"
build_args=(--build-arg "ASCEND_REPO=$ascend_repo"
            --build-arg "ASCEND_BRANCH=$ascend_branch")

# The corporate gateway inspects TLS, so pip/uv need its root CA. Pass it as a
# build arg; the Dockerfile installs it only when the value is non-empty.
ca_file="${CORPORATE_CA_FILE:-}"
if [[ -z "$ca_file" ]]; then
    for candidate in \
        /etc/pki/ca-trust/source/anchors/huawei-web-secure-gateway-ca-v2.pem \
        /etc/pki/ca-trust/source/anchors/huawei-bpit-root.pem \
        /usr/local/share/ca-certificates/huawei-web-secure-gateway-ca-v2.crt
    do
        if [[ -f "$candidate" ]]; then
            ca_file="$candidate"
            break
        fi
    done
fi

if [[ -n "$ca_file" ]]; then
    echo "Using corporate CA: $ca_file" >&2
    build_args+=(--build-arg "CORPORATE_CA=$(cat "$ca_file")")
else
    echo "No corporate CA found; the image will not trust a TLS-inspecting proxy." >&2
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec docker build --platform "$platform" --target "$target" \
    --build-arg "MAX_JOBS=${MAX_JOBS:-32}" \
    -f "$repo_dir/ci/Dockerfile" -t "$image" "$@" \
    "${build_args[@]}" "$repo_dir"
