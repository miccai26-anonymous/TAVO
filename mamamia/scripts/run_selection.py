#!/usr/bin/env python3
"""Run embedding-based selection methods.

Loads pool and query embeddings, runs specified selection method,
and saves results as JSON.

Usage:
    python run_selection.py \
        --method rds \
        --query-embeddings query.jsonl \
        --pool-embeddings pool.jsonl \
        --budget 250 \
        --output selections/rds_250.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from utils.data_loading import load_embeddings
from selection import SELECTION_METHODS, get_selection_method


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", type=str, required=True,
                        choices=list(SELECTION_METHODS.keys()))
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--pool-embeddings", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading embeddings...")
    pool_ids, pool_emb = load_embeddings(str(args.pool_embeddings))
    query_ids, query_emb = load_embeddings(str(args.query_embeddings))

    print(f"Pool: {len(pool_ids)}, Query: {len(query_ids)}")
    print(f"Method: {args.method}, Budget: {args.budget}")

    method = get_selection_method(args.method)(seed=args.seed)
    result = method.select(
        pool_ids=pool_ids, budget=args.budget,
        embeddings=pool_emb, query_embeddings=query_emb,
    )

    print(f"Selected {len(result.selected)} samples")

    output = {
        "method": result.method,
        "budget": result.budget,
        "selected": result.selected,
        "scores": result.scores,
        "metadata": result.metadata,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
