# SPDX-License-Identifier: Apache-2.0
"""Insert the project's SPDX header while preserving Python file preambles."""

import re
import sys
from pathlib import Path

HEADER = "# SPDX-License-Identifier: Apache-2.0\n"


def main() -> int:
    """Insert missing headers and report changes for pre-commit to review."""
    changed = False
    for filename in sys.argv[1:]:
        path = Path(filename)
        text = path.read_text(encoding="utf-8")
        if HEADER.strip() in text.splitlines()[:5]:
            continue
        lines = text.splitlines(keepends=True)
        index = int(bool(lines and lines[0].startswith("#!")))
        for position, line in enumerate(lines[:2]):
            if re.match(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+", line):
                index = position + 1
        lines.insert(index, HEADER)
        path.write_text("".join(lines), encoding="utf-8")
        print(f"Added SPDX header: {filename}")
        changed = True
    return int(changed)


if __name__ == "__main__":
    raise SystemExit(main())
