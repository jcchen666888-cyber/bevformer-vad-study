#!/usr/bin/env python3
"""Check that repository-local Markdown links and images resolve."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    markdown_files = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    failures: list[str] = []
    checked = 0
    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        for raw in LINK.findall(text):
            target = raw.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            target = unquote(target)
            resolved = (document.parent / target).resolve()
            checked += 1
            if not resolved.exists():
                failures.append(f"{document.relative_to(root)} -> {target}")
    if failures:
        raise SystemExit("Broken local links:\n" + "\n".join(failures))
    print(f"DOC_LINKS_OK: {checked} local links across {len(markdown_files)} files")


if __name__ == "__main__":
    main()
