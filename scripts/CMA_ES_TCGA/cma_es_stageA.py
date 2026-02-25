#!/usr/bin/env python3
import sys
import json
import numpy as np
import cma
from pathlib import Path

PROJECT_ROOT = Path("/path/to/project")
sys.path.append(str(PROJECT_ROOT))

from scripts.CMA_ES_TCGA.short_eval_weight import evaluate_weight_stable

# =========================
# Config (TCGA, no repeat)
# =========================
BUDGET_T = 10      # 5T or 10T
T = 50

SEEDS = "0,1,2"   # multi-seed in ONE run

POP_SIZE = 8
SIGMA0 = 0.2
N_GEN = 5

RUN_TAG = "stageA_iter400_gap_v7_new1"

OUT_ROOT = (
    PROJECT_ROOT
    / "outputs_TCGA_cma"
    / f"{BUDGET_T}T"
    / RUN_TAG
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_ROOT / "cma_es_log.json"


# =========================
# Utils
# =========================
def project_to_simplex(x):
    """
    Project R^2 -> simplex:
    w >= 0, sum = 1
    """
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s < 1e-8:
        return np.array([0.5, 0.5])
    return x / s



def fitness_fn(x):
    """
    CMA-ES objective:
    x -> projected weights -> short eval fitness
    CMA minimizes, so return -fitness (because our fitness is larger=better)
    """
    w = project_to_simplex(np.array(x))
    w_rds, w_less = float(w[0]), float(w[1])

    # sanity
    if not (0.0 <= w_rds <= 1.0 and 0.0 <= w_less <= 1.0):
        print(f"⚠️  [WARN] projected weights out of bounds: w_rds={w_rds}, w_less={w_less}")
    if abs((w_rds + w_less) - 1.0) > 1e-6:
        print(f"⚠️  [WARN] simplex sum != 1: sum={w_rds + w_less:.8f}")

    print(f"\n🔎 Eval: w_rds={w_rds:.4f}, w_less={w_less:.4f} (seeds={SEEDS})")

    fitness = evaluate_weight_stable(
        budget_T=BUDGET_T,
        w_rds=w_rds,
        w_less=w_less,
        run_tag=RUN_TAG,
        seeds=SEEDS,
    )

    print(f"🎯 Fitness = {fitness:.6f}")
    return -float(fitness)


# =========================
# Corner evaluation (inject)
# =========================
def eval_corners():
    corners = {
        "RDS_only": (1.0, 0.0),
        "LESS_only": (0.0, 1.0),
        "Equal": (0.5, 0.5),
    }

    results = {}
    print("\n==============================")
    print("🧪 Corner case evaluation")
    print("==============================")

    for name, (w1, w2) in corners.items():
        print(f"\n▶ {name}: w_rds={w1:.3f}, w_less={w2:.3f} (seeds={SEEDS})")

        f = evaluate_weight_stable(
            budget_T=BUDGET_T,
            w_rds=w1,
            w_less=w2,
            run_tag=RUN_TAG,
            seeds=SEEDS,
        )

        print(f"🎯 Corner fitness = {float(f):.6f}")
        results[name] = {
            "w_rds": float(w1),
            "w_less": float(w2),
            "fitness": float(f),
        }

    return results


# =========================
# Main
# =========================
def main():
    print("=" * 60)
    print("🚀 CMA-ES Stage A (TCGA, no repeat)")
    print(f"budget_T = {BUDGET_T}T")
    print(f"T        = {T}")
    print(f"seeds    = {SEEDS}")
    print(f"popsize  = {POP_SIZE}")
    print(f"sigma0   = {SIGMA0}")
    print(f"n_gen    = {N_GEN}")
    print(f"run_tag  = {RUN_TAG}")
    print("=" * 60)

    log = {
        "dataset": "TCGA",
        "budget_T": int(BUDGET_T),
        "T": int(T),
        "seeds": str(SEEDS),
        "popsize": int(POP_SIZE),
        "sigma0": float(SIGMA0),
        "generations": [],
        "corner_results": {},
    }

    # ---------- corners (and init incumbent from best corner) ----------
    log["corner_results"] = eval_corners()

    best_corner_name, best_corner_dict = max(
        log["corner_results"].items(), key=lambda kv: kv[1]["fitness"]
    )

    incumbent = {
        "w_rds": float(best_corner_dict["w_rds"]),
        "w_less": float(best_corner_dict["w_less"]),
        "fitness": float(best_corner_dict["fitness"]),
        "src": f"corner_{best_corner_name}",
        "seeds": str(SEEDS),
    }

    # ---------- CMA init: start from best corner (more stable than fixed 0.5/0.5) ----------
    x0 = np.array([incumbent["w_rds"], incumbent["w_less"]], dtype=np.float64)

    es = cma.CMAEvolutionStrategy(
        x0,
        SIGMA0,
        {
            "popsize": POP_SIZE,
            "bounds": [0.0, 1.0],
            "verb_log": 0,
            "verbose": -9,
        },
    )

    # ---------- generations ----------
    for gen in range(N_GEN):
        print(f"\n==============================")
        print(f"🚀 CMA-ES Generation {gen}")
        print("==============================")

        X = es.ask()
        fitness_vals = []

        gen_log = {
            "generation": int(gen),
            "candidates": [],
        }

        for x in X:
            # CMA asks in raw space
            f = fitness_fn(x)  # minimized value
            fitness_vals.append(float(f))

            w = project_to_simplex(x)
            cand_fit = -float(f)  # maximized fitness

            gen_log["candidates"].append(
                {
                    "raw_x": list(map(float, np.asarray(x).tolist())),
                    "w_rds": float(w[0]),
                    "w_less": float(w[1]),
                    "fitness": float(cand_fit),
                    "src": "cma",
                    "seeds": str(SEEDS),
                }
            )

            if cand_fit > incumbent["fitness"]:
                incumbent = {
                    "w_rds": float(w[0]),
                    "w_less": float(w[1]),
                    "fitness": float(cand_fit),
                    "src": "cma",
                    "seeds": str(SEEDS),
                }

        es.tell(X, fitness_vals)

        # logging: best from CMA internal (optional)
        best_x = np.asarray(es.result.xbest, dtype=np.float64)
        best_w = project_to_simplex(best_x)

        gen_log["best"] = {
            "raw_x": list(map(float, best_x.tolist())),
            "w_rds": float(best_w[0]),
            "w_less": float(best_w[1]),
            "fitness": float(-es.result.fbest),
        }

        log["generations"].append(gen_log)
        log["incumbent"] = incumbent
        log["final_safe_selection"] = incumbent  # keep same field name as C4

        print("\n🏆 Best CMA (this gen):")
        print(f"   w_rds={best_w[0]:.4f}, w_less={best_w[1]:.4f}")
        print(f"   fitness={-es.result.fbest:.6f}")
        print(
            f"🛡️  Incumbent (safe): "
            f"w_rds={incumbent['w_rds']:.4f}, "
            f"w_less={incumbent['w_less']:.4f}, "
            f"fitness={incumbent['fitness']:.6f}, "
            f"src={incumbent['src']}, "
            f"seeds={incumbent['seeds']}"
        )

        with open(LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)

    # ---------- Save Top-K for Stage B (include corners + CMA) ----------
    all_candidates = []

    # corners as candidates (INJECT)
    for name, d in log["corner_results"].items():
        all_candidates.append(
            {
                "w_rds": float(d["w_rds"]),
                "w_less": float(d["w_less"]),
                "fitness": float(d["fitness"]),
                "src": f"corner_{name}",
                "seeds": str(SEEDS),
            }
        )

    # CMA candidates
    for g in log["generations"]:
        all_candidates.extend(g["candidates"])

    all_candidates.sort(key=lambda x: x["fitness"], reverse=True)
    top10 = all_candidates[:10]

    with open(OUT_ROOT / "stageA_top10.json", "w") as f:
        json.dump(top10, f, indent=2)

    print("\n🎉 CMA-ES Stage A finished (TCGA)")
    print(f"📄 Log saved to {LOG_PATH}")
    print(f"📄 Top-10 saved to {OUT_ROOT / 'stageA_top10.json'}")
    print("\n🏁 Final SAFE selection:")
    print(
        f"   w_rds={incumbent['w_rds']:.4f}, "
        f"w_less={incumbent['w_less']:.4f}, "
        f"fitness={incumbent['fitness']:.6f}, "
        f"src={incumbent['src']}, "
        f"seeds={incumbent['seeds']}"
    )


if __name__ == "__main__":
    main()
