#!/usr/bin/env python3
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/path/to/project")
sys.path.append(str(PROJECT_ROOT))

from scripts.CMA_ES_C4.short_eval_weight import evaluate_weight_stageB

# --------------------------------------------------
# Config
# --------------------------------------------------
REPEAT_ID = 1
BUDGET_T = 5

RUN_TAG_A = "stageA_iter400_gap_v7"
RUN_TAG_B = "stageB_iter2000_clean_v7"

TOPK_PATH = (
    PROJECT_ROOT / "outputs_C4_cma"
    / f"repeat{REPEAT_ID:02d}"
    / f"{BUDGET_T}T"
    / RUN_TAG_A
    / "stageA_top10.json"
)

OUT_DIR = (
    PROJECT_ROOT / "outputs_C4_cma"
    / f"repeat{REPEAT_ID:02d}"
    / f"{BUDGET_T}T"
    / RUN_TAG_B
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "stageB_results.json"

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    with open(TOPK_PATH) as f:
        top10 = json.load(f)

    results = []

    print("\n==============================")
    print("🔁 Stage B | Re-ranking Top-10")
    print("==============================")
    print(f"📥 From Stage A: {RUN_TAG_A}")
    print(f"📤 Writing Stage B results to: {RUN_TAG_B}")

    for item in top10:
        w_rds = float(item["w_rds"])
        w_less = float(item["w_less"])

        print(f"\n▶ wR={w_rds:.3f}, wL={w_less:.3f}")

        f = evaluate_weight_stageB(
            repeat_id=REPEAT_ID,
            budget_T=BUDGET_T,
            w_rds=w_rds,
            w_less=w_less,
            run_tag=RUN_TAG_B,
        )

        results.append({
            "w_rds": w_rds,
            "w_less": w_less,
            "fitness": f,
        })

    results.sort(key=lambda x: x["fitness"], reverse=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Stage B finished")
    print(f"📄 Results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
