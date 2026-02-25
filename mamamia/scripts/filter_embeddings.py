#!/usr/bin/env python3
"""Filter embeddings JSONL to a subset of case IDs.

Usage:
    python filter_embeddings.py \
        --input all_embeddings.jsonl \
        --output filtered.jsonl \
        --case-ids CASE_01 CASE_02
    # or
    python filter_embeddings.py \
        --input all_embeddings.jsonl \
        --output filtered.jsonl \
        --case-list cases.txt
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-ids", type=str, nargs="+", default=None)
    parser.add_argument("--case-list", type=Path, default=None)
    args = parser.parse_args()

    if args.case_list and args.case_list.exists():
        case_ids = set(l.strip() for l in args.case_list.read_text().splitlines() if l.strip())
    elif args.case_ids:
        case_ids = set(args.case_ids)
    else:
        raise ValueError("Must provide --case-ids or --case-list")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.input.open("r") as f_in, args.output.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["image"] in case_ids:
                f_out.write(line)
                kept += 1

    print(f"Filtered: {kept}/{len(case_ids)} cases written to {args.output}")


if __name__ == "__main__":
    main()
