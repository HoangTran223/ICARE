#!/usr/bin/env python3
"""One-shot patch: remove machine-specific BASE_PATH from scripts/*.sh."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

SETUP_BLOCK = """SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../common/setup_env.sh
source "${SCRIPT_DIR}/../common/setup_env.sh"
"""

HARD_BASE_RE = re.compile(
    r"^BASE_PATH=(/home/hungpv[^\s]*|/mnt/hungpv[^\s]*|/project/hungpv[^\s]*|path_to_project)\s*$"
)


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "common" in path.parts and path.name == "setup_env.sh":
        return False

    lines = text.split("\n")
    out: list[str] = []
    has_setup = "setup_env.sh" in text
    inserted = has_setup
    changed = False

    for line in lines:
        if HARD_BASE_RE.match(line.strip()):
            if not inserted:
                out.extend(SETUP_BLOCK.split("\n"))
                inserted = True
                changed = True
            changed = True
            continue
        out.append(line)

    new_text = "\n".join(out)
    if not has_setup and 'BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"' in new_text:
        repl = (
            'BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"\n'
            "# shellcheck source=../common/setup_env.sh\n"
            'source "${SCRIPT_DIR}/../common/setup_env.sh"'
        )
        if repl not in new_text:
            new_text = new_text.replace(
                'BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"',
                repl,
                1,
            )
            changed = True

    if changed and new_text != text:
        path.write_text(new_text + ("\n" if not new_text.endswith("\n") else ""), encoding="utf-8")
        return True
    return False


def main() -> None:
    n = 0
    for sh in sorted(SCRIPTS.rglob("*.sh")):
        if sh.parent.name == "common":
            continue
        if patch_file(sh):
            print(f"patched {sh.relative_to(ROOT)}")
            n += 1
    print(f"done: {n} files patched")


if __name__ == "__main__":
    main()
