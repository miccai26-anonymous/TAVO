#!/usr/bin/env python3
import os
import sys
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 修正 PYTHONPATH，方便以后如果要 import 别的模块
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f">>> PROJECT_ROOT = {PROJECT_ROOT}")
print(f">>> sys.path[0] = {sys.path[0]}")

# ============================================================
# 配置路径（根据你现在的结构）
# ============================================================
BASE_DIR = PROJECT_ROOT   # = .../EfficientVit
GRAD_DIR = os.path.join(BASE_DIR, "results/orient_gradients")
SUBSET_ROOT = os.path.join(BASE_DIR, "data/splits_21_orient_static_T60")

# ============================================================
# 1. 加载 case-level 向量和 ID
# ============================================================
src_vecs = np.load(os.path.join(GRAD_DIR, "src_case_vecs.npy"))
tgt_vecs = np.load(os.path.join(GRAD_DIR, "tgt_case_vecs.npy"))

with open(os.path.join(GRAD_DIR, "src_case_ids.txt")) as f:
    src_ids = [l.strip() for l in f if l.strip()]

with open(os.path.join(GRAD_DIR, "tgt_case_ids.txt")) as f:
    tgt_ids = [l.strip() for l in f if l.strip()]

Ns, D_s = src_vecs.shape
Nt, D_t = tgt_vecs.shape

print(f"Loaded src cases: {len(src_ids)}, tgt cases: {len(tgt_ids)}")
print(f"src_vecs shape = {src_vecs.shape}, tgt_vecs shape = {tgt_vecs.shape}")

assert Ns == len(src_ids), "❌ src_vecs rows != src_ids count"
assert Nt == len(tgt_ids), "❌ tgt_vecs rows != tgt_ids count"

# ============================================================
# 2. 计算 naive 的 “与 target 的平均 cosine 相似度”
#    sim_mean[i] = mean_j cos(src_i, tgt_j)
# ============================================================
print("\n📐 Computing cosine similarity matrix src × tgt ...")
sim_matrix = cosine_similarity(src_vecs, tgt_vecs)  # (Ns, Nt)

sim_mean = sim_matrix.mean(axis=1)   # (Ns,)
# 你也可以改成 sim_max = sim_matrix.max(axis=1)，看哪个更稳

# 排名：越大越前 → rank=1 是最相似
order_desc = np.argsort(-sim_mean)              # indices sorted by descending sim
ranks = np.empty_like(order_desc)
ranks[order_desc] = np.arange(1, Ns + 1)        # ranks[i] = 1..Ns

print("✅ Finished computing mean similarity + ranks.")
print(f"Example: best src index = {order_desc[0]} with sim_mean = {sim_mean[order_desc[0]]:.4f}")

# ============================================================
# 3. 封装一个小工具：给定 ORIENT subset，做 rank 分析
# ============================================================
def analyze_subset(tag, selected_ids):
    """
    tag: 例如 'orient_1T'
    selected_ids: list of subject IDs (e.g. 'BraTS2021_00001')
    """
    id_to_idx = {sid: i for i, sid in enumerate(src_ids)}

    sel_indices = []
    missing = []
    for sid in selected_ids:
        if sid in id_to_idx:
            sel_indices.append(id_to_idx[sid])
        else:
            missing.append(sid)

    sel_indices = np.array(sel_indices, dtype=int)

    print(f"\n==== 🔎 Analysis for {tag} ====")
    print(f"#selected_ids (file): {len(selected_ids)}")
    print(f"#mapped to src_vecs : {len(sel_indices)}")
    if missing:
        print(f"⚠️  {len(missing)} IDs not found in src_ids (show first 5): {missing[:5]}")

    sel_ranks = ranks[sel_indices]  # 越小越好
    sel_sims  = sim_mean[sel_indices]

    # 基本统计
    print(f"Rank stats (1 = best, Ns = worst, Ns = {Ns}):")
    print(f"  min rank  = {sel_ranks.min()}")
    print(f"  10% perc  = {np.percentile(sel_ranks, 10):.1f}")
    print(f"  25% perc  = {np.percentile(sel_ranks, 25):.1f}")
    print(f"  median    = {np.median(sel_ranks):.1f}")
    print(f"  75% perc  = {np.percentile(sel_ranks, 75):.1f}")
    print(f"  90% perc  = {np.percentile(sel_ranks, 90):.1f}")
    print(f"  max rank  = {sel_ranks.max()}")

    # 覆盖 top-k (top10%, 20%, 50%)
    def coverage(top_frac):
        cutoff = int(np.ceil(Ns * top_frac))
        return (sel_ranks <= cutoff).mean(), cutoff

    for frac in [0.1, 0.2, 0.5]:
        cov, cutoff = coverage(frac)
        print(f"  % in top {int(frac*100)}% (rank ≤ {cutoff}): {cov*100:.2f}%")

    # 输出前 10 个 ORIENT 选中的 case，看它们的 naive 排名和 similarity
    print("\n  Top 10 ORIENT-selected (by naive similarity within subset):")
    idx_sorted_inside = sel_indices[np.argsort(-sel_sims)]  # 子集内部按 sim 排序
    for i in idx_sorted_inside[:10]:
        print(
            f"    ID={src_ids[i]:<25}  "
            f"global_rank={ranks[i]:4d}  "
            f"mean_sim={sim_mean[i]:.4f}"
        )


# ============================================================
# 4. 对 4 个 subset 做分析
# ============================================================
def main():
    T = 50
    budgets_T = [1, 5, 10, 15]

    for k in budgets_T:
        tag = f"orient_{k}T"
        subset_dir = os.path.join(SUBSET_ROOT, tag)
        subset_file = os.path.join(subset_dir, "train_subjects.txt")

        if not os.path.exists(subset_file):
            print(f"\n❌ {tag}: subset file not found: {subset_file}")
            continue

        with open(subset_file) as f:
            selected_ids = [l.strip() for l in f if l.strip()]

        analyze_subset(tag, selected_ids)


if __name__ == "__main__":
    main()
