"""Extract distinct career description strings from the candidate pool and count frequencies.

This is the script that discovered the 44-template phenomenon: running it
early on revealed the pool's descriptions collapse to a small, fixed set of
canonical strings. The output feeds the template audit and calibration scripts.

Usage:
    python tools/extract_career_templates.py --candidates ./dataset/candidates.jsonl \
        --out ./dataset/career_description_templates.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def extract_unique_templates(candidates_path: Path) -> list[dict]:
    """Scan the candidate pool and return unique description templates with counts."""
    frequency_counter: collections.Counter[str] = collections.Counter()
    with open(candidates_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            for role in record.get("career_history", []):
                description = role.get("description")
                if description:
                    frequency_counter[description] += 1

    ordered = sorted(frequency_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {"id": index, "count": count, "text": text}
        for index, (text, count) in enumerate(ordered)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    templates = extract_unique_templates(args.candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file_handle:
        json.dump(templates, file_handle, ensure_ascii=False, indent=2)

    print(f"Extracted {len(templates)} distinct templates -> {args.out}")
    for template in templates:
        preview = template["text"][:60].replace("\n", " ")
        print(f"  T{template['id']:02d}  {template['count']:>6}x  {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
