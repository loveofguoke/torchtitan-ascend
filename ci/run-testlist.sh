#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run only the entries listed in ci/testlists/<target>.txt.
set -euo pipefail

usage() {
    echo "Usage: $0 [cpu|a2] [extra pytest arguments...]"
    echo 'Environment: WORKDIR (default /opt/torchtitan), BATCH (default 150)'
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
    cpu|a2) ;;
    *) usage >&2; exit 2 ;;
esac
BATCH="${BATCH:-150}"
if [[ ! "$BATCH" =~ ^[1-9][0-9]*$ ]]; then
    echo 'ERROR: BATCH must be a positive integer' >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST="$HERE/testlists/$TARGET.txt"
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

failed_files="$PARTS_DIR/failures.log"
: > "$failed_files"
split -l "$BATCH" "$CASES" "$PARTS_DIR/part."

for part in "$PARTS_DIR"/part.*; do
    batch=()
    while IFS= read -r one; do
        batch+=("$one")
    done < "$part"
    echo "[run-testlist] running ${#batch[@]} entries"
    if pytest "$@" -vv -rs "${batch[@]}"; then
        :
    else
        echo '[run-testlist] batch had failures; re-running case-by-case to isolate them'
        for one in "${batch[@]}"; do
            if ! pytest "$@" -q "$one"; then
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
