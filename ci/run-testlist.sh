#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run only the entries listed in ci/testlists/<target>.txt.
set -euo pipefail

usage() {
    echo "Usage: $0 [cpu|a2-1card|a2-2card|a2-8card] [extra pytest arguments...]"
    echo 'Environment: WORKDIR (default /opt/torchtitan), BATCH (default 150)'
    echo '             TESTLIST (default ci/testlists/<target>.txt; relative to the invocation directory)'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi

TARGET="${1:-cpu}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$TARGET" in
    cpu|a2-1card|a2-2card|a2-8card) ;;
    *) usage >&2; exit 2 ;;
esac
BATCH="${BATCH:-150}"
if [[ ! "$BATCH" =~ ^[1-9][0-9]*$ ]]; then
    echo 'ERROR: BATCH must be a positive integer' >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST="${TESTLIST:-$HERE/testlists/$TARGET.txt}"
if [[ ! -f "$LIST" ]]; then
    echo "ERROR: test list not found: $LIST" >&2
    exit 2
fi

CASES="$(mktemp)"
PARTS_DIR="$(mktemp -d)"
trap 'rm -rf "$CASES" "$PARTS_DIR"' EXIT
sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "$LIST" > "$CASES"
COUNT=$(wc -l < "$CASES")
if [[ "$COUNT" -eq 0 ]]; then
    echo "ERROR: no test cases in $LIST" >&2
    exit 2
fi
echo "[run-testlist] target=$TARGET entries=$COUNT"

WORKDIR="${WORKDIR:-/opt/torchtitan}"
cd "$WORKDIR" || { echo "ERROR: workdir $WORKDIR not found" >&2; exit 2; }

source_dir="$(cd "$HERE/.." && pwd)"
# Suppress the old NPU startup hook only while installing the checkout.
PYTHONPATH= TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pip install --no-deps -e "$source_dir"
if [[ "$TARGET" == a2* ]]; then
    cp -a "$source_dir/tests/." ./tests/
else
    cp -a "$source_dir/tests/unit_tests/cpu/." ./tests/unit_tests/cpu/
fi

# Apply patches in the same Python process that collects and runs the tests.
run_pytest() {
    python -c '
from torchtitan_ascend import setup
setup()
import pytest
import sys
sys.exit(pytest.main(sys.argv[1:]))
' "$@"
}

failed_files="$PARTS_DIR/failures.log"
: > "$failed_files"
split -l "$BATCH" "$CASES" "$PARTS_DIR/part."

for part in "$PARTS_DIR"/part.*; do
    batch=()
    while IFS= read -r one; do
        batch+=("$one")
    done < "$part"
    echo "[run-testlist] running ${#batch[@]} entries"
    if run_pytest "$@" -vv -rs "${batch[@]}"; then
        :
    else
        echo '[run-testlist] batch had failures; re-running case-by-case to isolate them'
        for one in "${batch[@]}"; do
            if ! run_pytest "$@" -q "$one"; then
                echo "FAILED_OR_ERROR $one" | tee -a "$failed_files"
            fi
        done
    fi
done

if [[ -s "$failed_files" ]]; then
    echo
    echo '[run-testlist] ===== entries with pytest failures or errors ====='
    cat "$failed_files"
    exit 1
fi

echo "[run-testlist] completed $COUNT list entries without pytest failures; check pytest summaries for skips"
