import os
import pandas as pd
import numpy as np

INPUT_PATH = "results/decision_layer_results.csv"
OUTPUT_PATH = "results/human_in_the_loop_results.csv"
SUMMARY_PATH = "results/human_in_the_loop_summary.csv"

def main():
    print("=== Human-in-the-Loop Decision Analysis ===")

    df = pd.read_csv(INPUT_PATH)

    required = ["risk_level", "recommended_action", "sap_action"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Simulated human review policy.
    # High-risk cases are reviewed.
    # A small fraction of recommendations are overridden.
    rng = np.random.default_rng(42)

    df["human_review_required"] = df["risk_level"].isin(
        ["MEDIUM", "HIGH"]
    )

    df["human_decision"] = df["sap_action"]

    review_mask = df["human_review_required"]

    override_probability = np.where(
        df["risk_level"].eq("HIGH"),
        0.15,
        0.08,
    )

    override_mask = (
        review_mask
        & (rng.random(len(df)) < override_probability)
    )

    df.loc[override_mask, "human_decision"] = (
        "HUMAN_OVERRIDE_REVIEW"
    )

    df["decision_status"] = np.where(
        override_mask,
        "OVERRIDDEN",
        np.where(
            review_mask,
            "HUMAN_APPROVED",
            "AUTOMATED",
        ),
    )

    df["final_action"] = df["human_decision"]

    os.makedirs("results", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    summary = (
        df.groupby(
            ["risk_level", "decision_status"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    summary["percentage"] = (
        summary["count"] / len(df) * 100
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\n=== Risk Distribution ===")
    print(df["risk_level"].value_counts())

    print("\n=== Human Review ===")
    print(df["decision_status"].value_counts())

    print("\n=== Override Rate ===")
    print(
        f"{override_mask.mean() * 100:.2f}%"
    )

    print("\n=== Decision Summary ===")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(f"- {OUTPUT_PATH}")
    print(f"- {SUMMARY_PATH}")


if __name__ == "__main__":
    main()