from pathlib import Path
import json
import pandas as pd

RESULTS = Path("results")


def pct(x):
    return round(float(x) * 100, 2)


def main():
    print("=== Paper Results Summary ===")

    summary = {}

    # -------------------------
    # Model performance
    # -------------------------
    df = pd.read_csv(RESULTS / "experiment_summary.csv")

    roc = df[df["metric"] == "ROC-AUC"].copy()
    pr = df[df["metric"] == "PR-AUC"].copy()

    summary["model_performance"] = {
        "best_roc_auc_model": (
            roc.loc[roc["mean"].idxmax(), "model"]
            if not roc.empty else None
        ),
        "best_roc_auc": (
            float(roc["mean"].max())
            if not roc.empty else None
        ),
        "best_pr_auc_model": (
            pr.loc[pr["mean"].idxmax(), "model"]
            if not pr.empty else None
        ),
        "best_pr_auc": (
            float(pr["mean"].max())
            if not pr.empty else None
        ),
    }

    # -------------------------
    # Decision effectiveness
    # -------------------------
    decision = pd.read_csv(
        RESULTS / "decision_effectiveness.csv"
    )

    low = decision[
        decision["risk_level"].str.upper() == "LOW"
    ].iloc[0]

    high = decision[
        decision["risk_level"].str.upper() == "HIGH"
    ].iloc[0]

    summary["decision_effectiveness"] = {
        "low_late_rate": float(low["observed_late_rate"]),
        "high_late_rate": float(high["observed_late_rate"]),
        "absolute_risk_gap": float(
            high["observed_late_rate"]
            - low["observed_late_rate"]
        ),
        "relative_increase": float(
            high["observed_late_rate"]
            / low["observed_late_rate"]
            - 1
        ),
        "monotonic_risk_order": bool(
            decision["monotonic_risk_order"].iloc[0]
        ),
        "chi_square": float(
            decision["chi_square"].iloc[0]
        ),
        "chi_square_p_value": float(
            decision["chi_square_p_value"].iloc[0]
        ),
    }

    # -------------------------
    # Calibration
    # -------------------------
    calibration_summary = pd.read_csv(
        RESULTS / "calibration_summary.csv"
    )

    summary["calibration"] = {
        column: (
            float(calibration_summary[column].iloc[0])
            if pd.api.types.is_numeric_dtype(
                calibration_summary[column]
            )
            else str(calibration_summary[column].iloc[0])
        )
        for column in calibration_summary.columns
    }

    # -------------------------
    # Economic evaluation
    # -------------------------
    economic = pd.read_csv(
        RESULTS / "end_to_end_economic_evaluation.csv"
    )

    total_baseline = economic["baseline_cost"].sum()
    total_decision = economic["decision_cost"].sum()
    total_saving = economic["cost_saving"].sum()

    summary["economic_impact"] = {
        "baseline_cost": float(total_baseline),
        "decision_cost": float(total_decision),
        "cost_saving": float(total_saving),
        "saving_rate": float(
            total_saving / total_baseline
        ),
    }

    # -------------------------
    # Economic sensitivity
    # -------------------------
    sensitivity = pd.read_csv(
        RESULTS / "economic_scenario_analysis.csv"
    )

    beneficial = sensitivity[
        sensitivity["economically_beneficial"] == True
    ]

    summary["economic_sensitivity"] = {
        "beneficial_scenarios": int(len(beneficial)),
        "total_scenarios": int(len(sensitivity)),
        "beneficial_percentage": float(
            len(beneficial) / len(sensitivity)
        ),
    }

    # -------------------------
    # Break-even
    # -------------------------
    breakeven = pd.read_csv(
        RESULTS / "economic_break_even.csv"
    )

    summary["break_even"] = breakeven.to_dict(
        orient="records"
    )

    # -------------------------
    # Human-in-the-loop
    # -------------------------
    hitl_path = RESULTS / "human_in_the_loop.csv"

    if hitl_path.exists():
        hitl = pd.read_csv(hitl_path)

        summary["human_in_the_loop"] = (
            hitl.to_dict(orient="records")
        )

    # -------------------------
    # Save JSON
    # -------------------------
    json_path = RESULTS / "paper_results_summary.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    # -------------------------
    # Generate Markdown
    # -------------------------
    lines = [
        "# Paper Results Summary",
        "",
        "## Model Performance",
        "",
        f"- Best ROC-AUC model: "
        f"{summary['model_performance']['best_roc_auc_model']}",
        f"- Best ROC-AUC: "
        f"{summary['model_performance']['best_roc_auc']:.4f}",
        f"- Best PR-AUC model: "
        f"{summary['model_performance']['best_pr_auc_model']}",
        f"- Best PR-AUC: "
        f"{summary['model_performance']['best_pr_auc']:.4f}",
        "",
        "## Decision Effectiveness",
        "",
        f"- LOW observed late rate: "
        f"{pct(summary['decision_effectiveness']['low_late_rate']):.2f}%",
        f"- HIGH observed late rate: "
        f"{pct(summary['decision_effectiveness']['high_late_rate']):.2f}%",
        f"- Absolute risk gap: "
        f"{pct(summary['decision_effectiveness']['absolute_risk_gap']):.2f} percentage points",
        f"- Relative increase: "
        f"{pct(summary['decision_effectiveness']['relative_increase']):.2f}%",
        f"- Monotonic risk ordering: "
        f"{summary['decision_effectiveness']['monotonic_risk_order']}",
        f"- Chi-square: "
        f"{summary['decision_effectiveness']['chi_square']:.4f}",
        f"- p-value: "
        f"{summary['decision_effectiveness']['chi_square_p_value']:.6g}",
        "",
        "## Economic Impact",
        "",
        f"- Baseline cost: "
        f"${summary['economic_impact']['baseline_cost']:,.2f}",
        f"- Decision cost: "
        f"${summary['economic_impact']['decision_cost']:,.2f}",
        f"- Cost saving: "
        f"${summary['economic_impact']['cost_saving']:,.2f}",
        f"- Saving rate: "
        f"{pct(summary['economic_impact']['saving_rate']):.2f}%",
        "",
        "## Economic Sensitivity",
        "",
        f"- Beneficial scenarios: "
        f"{summary['economic_sensitivity']['beneficial_scenarios']}/"
        f"{summary['economic_sensitivity']['total_scenarios']}",
        f"- Beneficial percentage: "
        f"{pct(summary['economic_sensitivity']['beneficial_percentage']):.2f}%",
        "",
        "## Break-Even Analysis",
        "",
        "See `table_break_even.csv` and "
        "`table_break_even.tex` for the complete results.",
        "",
        "## Interpretation",
        "",
        "The results indicate whether the proposed SAP-oriented "
        "decision layer provides predictive separation, calibrated "
        "risk estimates, human-in-the-loop governance, and economic "
        "benefit under the tested cost assumptions.",
    ]

    md_path = RESULTS / "paper_results_summary.md"
    md_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\nSaved:")
    print(f"- {json_path}")
    print(f"- {md_path}")


if __name__ == "__main__":
    main()