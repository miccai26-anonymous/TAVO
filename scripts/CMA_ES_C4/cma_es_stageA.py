#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import cma
from pathlib import Path

# --------------------------------------------------
# Project imports
# --------------------------------------------------
PROJECT_ROOT = Path("/path/to/project")
sys.path.append(str(PROJECT_ROOT))

from scripts.CMA_ES_C4.short_eval_weight import evaluate_weight_stable

# --------------------------------------------------
# Config
# --------------------------------------------------
REPEAT_ID = 1
BUDGET_T = 10          # 5T or 10T
T = 50

SEEDS = "0,1,2"   # NEW: multi-seed in ONE run

POP_SIZE = 8
SIGMA0 = 0.2
N_GEN = 5

RUN_TAG = "stageA_iter400_gap_v7"

OUT_ROOT = (
    PROJECT_ROOT
    / "outputs_C4_cma"
    / f"repeat{REPEAT_ID:02d}"
    / f"{BUDGET_T}T"
    / RUN_TAG
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_ROOT / "cma_es_log.json"

# --------------------------------------------------
# Utils
# --------------------------------------------------
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
    CMA minimizes, so return -fitness
    """
    w = project_to_simplex(np.array(x))
    w_rds, w_less = float(w[0]), float(w[1])

    print(f"\n🔎 Eval weight: w_rds={w_rds:.4f}, w_less={w_less:.4f}")

    fitness = evaluate_weight_stable(
        repeat_id=REPEAT_ID,
        budget_T=BUDGET_T,
        w_rds=w_rds,
        w_less=w_less,
        run_tag=RUN_TAG,
        seeds=SEEDS,
    )

    print(f"🎯 Fitness = {fitness:.6f}")
    return -fitness


# --------------------------------------------------
# Corner evaluation (no CMA update)
# --------------------------------------------------
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
        print(f"\n▶ {name}: w_rds={w1}, w_less={w2}")

        f = evaluate_weight_stable(
            repeat_id=REPEAT_ID,
            budget_T=BUDGET_T,
            w_rds=w1,
            w_less=w2,
            run_tag=RUN_TAG,
            seeds=SEEDS,
        )

        print(f"🎯 Corner fitness = {f:.6f}")
        results[name] = {
            "w_rds": w1,
            "w_less": w2,
            "fitness": f,
        }

    return results


# --------------------------------------------------
# Main CMA-ES
# --------------------------------------------------
def main():

    log = {
        "repeat": REPEAT_ID,
        "budget_T": BUDGET_T,
        "popsize": POP_SIZE,
        "sigma0": SIGMA0,
        "generations": [],
        "corner_results": {},
    }

    # ---------- corner ----------
    log["corner_results"] = eval_corners()
    
    # ---------- init incumbent from corner ----------
    corner_best_item = max(
        log["corner_results"].items(),
        key=lambda x: x[1]["fitness"]
    )

    incumbent = {
        "w_rds": corner_best_item[1]["w_rds"],
        "w_less": corner_best_item[1]["w_less"],
        "fitness": corner_best_item[1]["fitness"],
        "src": f"corner_{corner_best_item[0]}",
    }

    # ---------- CMA init ----------
    x0 = np.array([0.5, 0.5])
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
            "generation": gen,
            "candidates": [],
        }

        for x in X:
            f = fitness_fn(x)
            fitness_vals.append(f)

            w = project_to_simplex(np.array(x))
            gen_log["candidates"].append({
                "raw_x": x.tolist(),
                "w_rds": float(w[0]),
                "w_less": float(w[1]),
                "fitness": -f,
            })
            cand_fitness = -f  # real (maximized) fitness

            if cand_fitness > incumbent["fitness"]:
                incumbent = {
                    "w_rds": float(w[0]),
                    "w_less": float(w[1]),
                    "fitness": cand_fitness,
                    "src": "cma",
                }

        es.tell(X, fitness_vals)

        best_x = es.result.xbest
        best_w = project_to_simplex(best_x)

        gen_log["best"] = {
            "w_rds": float(best_w[0]),
            "w_less": float(best_w[1]),
            "fitness": float(-es.result.fbest),
        }

        log["generations"].append(gen_log)
        log["final_safe_selection"] = incumbent

        print(f"\n🏆 Best so far:")
        print(f"   w_rds={best_w[0]:.4f}, w_less={best_w[1]:.4f}")
        print(f"   fitness={-es.result.fbest:.6f}")
        print(
        f"🛡️  Incumbent (safe): "
        f"w_rds={incumbent['w_rds']:.4f}, "
        f"w_less={incumbent['w_less']:.4f}, "
        f"fitness={incumbent['fitness']:.6f}, "
        f"src={incumbent['src']}"
)
        with open(LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)

    print("\n🎉 CMA-ES finished!")
    print(f"📄 Log saved to {LOG_PATH}")
    print("\n🏁 Final SAFE selection:")
    print(
        f"   w_rds={incumbent['w_rds']:.4f}, "
        f"w_less={incumbent['w_less']:.4f}, "
        f"fitness={incumbent['fitness']:.6f}, "
        f"src={incumbent['src']}"
    )


    # --------- Save Top-K for Stage B ---------
    all_candidates = []
    for g in log["generations"]:
        all_candidates.extend(g["candidates"])

    all_candidates.sort(key=lambda x: x["fitness"], reverse=True)
    top10 = all_candidates[:10]

    with open(OUT_ROOT / "stageA_top10.json", "w") as f:
        json.dump(top10, f, indent=2)

    print("✅ Stage A finished")
    print(f"📄 Top-10 saved to {OUT_ROOT / 'stageA_top10.json'}")


if __name__ == "__main__":
    main()
