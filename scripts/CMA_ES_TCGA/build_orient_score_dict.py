#!/usr/bin/env python3
import os
import argparse
import numpy as np

from submodlib.functions.facilityLocationMutualInformation import (
    FacilityLocationMutualInformationFunction
)
from sklearn.metrics.pairwise import cosine_similarity


def load_case_embeddings(embed_root, prefix):
    vecs = np.load(os.path.join(embed_root, f"{prefix}_case_vecs.npy"))
    with open(os.path.join(embed_root, f"{prefix}_case_ids.txt")) as f:
        ids = [line.strip() for line in f]
    assert vecs.shape[0] == len(ids)
    return ids, vecs


def orient_full_greedy(src_vecs, tgt_vecs, eta=1.0):
    Ns = src_vecs.shape[0]
    Nt = tgt_vecs.shape[0]

    # cosine similarities
    K = np.maximum(cosine_similarity(src_vecs, src_vecs), 0).astype(np.float64)
    Q = np.maximum(cosine_similarity(src_vecs, tgt_vecs), 0).astype(np.float64)

    obj = FacilityLocationMutualInformationFunction(
        n=Ns,
        num_queries=Nt,
        data_sijs=K,
        query_sijs=Q,
        magnificationEta=eta,
    )

    # full greedy: budget = Ns
    # 🔥 FIX: budget must be < effective ground set
    budget = Ns - 1

    result = obj.maximize(
        budget=budget,
        optimizer="LazyGreedy",
        stopIfNegativeGain=True,
        show_progress=True,
    )

    # preserve greedy order
    ordered_idx = []
    for elem in result:
        if isinstance(elem, tuple):
            ordered_idx.append(elem[0])
        else:
            ordered_idx.append(elem)

    return ordered_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=1.0)
    args = parser.parse_args()

    embed_root = f"results/orient_embeddings_TCGA"
    out_dir = f"data/splits_TCGA_orient"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Loading ORIENT embeddings from: {embed_root}")

    src_ids, src_vecs = load_case_embeddings(embed_root, "src")
    tgt_ids, tgt_vecs = load_case_embeddings(embed_root, "tgt")

    print(f"  Source subjects: {len(src_ids)}")
    print(f"  Target subjects: {len(tgt_ids)}")

    # --------------------------------------------------
    # Run ORIENT full greedy (NO gradient computation)
    # --------------------------------------------------
    print("\n🧭 Running ORIENT full greedy selection...")
    ordered_idx = orient_full_greedy(src_vecs, tgt_vecs, eta=args.eta)

    N = len(src_ids)
    orient_score_dict = {sid: 0.0 for sid in src_ids}

    for rank, idx in enumerate(ordered_idx):
        sid = src_ids[idx]
        orient_score_dict[sid] = float(N - rank)

    out_path = os.path.join(out_dir, "orient_score_dict.npy")
    np.save(out_path, orient_score_dict, allow_pickle=True)

    print(f"\n💾 Saved ORIENT score dict → {out_path}")
    print(f"   Num subjects: {len(orient_score_dict)}")


if __name__ == "__main__":
    main()
