from pathlib import Path
import pandas as pd

RESULTS = Path("results")


def save_table(df, name):
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    (RESULTS / f"{name}.tex").write_text(
        df.to_latex(index=False, float_format="%.4f"),
        encoding="utf-8"
    )


def main():
    print("=== Paper Results Tables ===")

    # 1. Model performance
    model_tables = []

    cv = pd.read_csv(RESULTS / "experiment_summary.csv")

    selected = cv[
        cv["metric"].isin(["ROC-AUC", "PR-AUC"])
    ].copy()

    if not selected.empty:
        model_tables.append(selected)

    if model_tables:
        models = pd.concat(model_tables, ignore_index=True)
        save_table(models, "table_model_performance")

    # 2. Decision effectiveness
    decision = pd.read_csv(
        RESULTS / "decision_effectiveness.csv"
    )

    decision_cols = [
        "risk_level",
        "orders",
        "mean_risk",
        "observed_late_rate",
        "risk_gap_high_vs_low",
        "relative_high_vs_low",
        "monotonic_risk_order",
        "chi_square",
        "chi_square_p_value",
    ]

    decision_cols = [
        c for c in decision_cols
        if c in decision.columns
    ]

    save_table(
        decision[decision_cols],
        "table_decision_effectiveness"
    )

    # 3. Calibration
    calibration = pd.read_csv(
        RESULTS / "calibration_analysis.csv"
    )

    save_table(
        calibration,
        "table_calibration"
    )

    # 4. Economic evaluation
    economic = pd.read_csv(
        RESULTS / "end_to_end_economic_evaluation.csv"
    )

    economic_cols = [
        "risk_level",
        "orders",
        "mean_probability",
        "baseline_cost",
        "decision_cost",
        "cost_saving",
        "saving_rate",
        "human_review_orders",
    ]

    economic_cols = [
        c for c in economic_cols
        if c in economic.columns
    ]

    save_table(
        economic[economic_cols],
        "table_economic_impact"
    )

    # 5. Economic scenarios
    scenarios = pd.read_csv(
        RESULTS / "economic_scenario_analysis.csv"
    )

    scenario_cols = [
        "late_delivery_cost",
        "intervention_cost",
        "cost_saving",
        "saving_rate",
        "economically_beneficial",
    ]

    scenario_cols = [
        c for c in scenario_cols
        if c in scenarios.columns
    ]

    save_table(
        scenarios[scenario_cols],
        "table_economic_sensitivity"
    )

    # 6. Break-even analysis
    breakeven = pd.read_csv(
        RESULTS / "economic_break_even.csv"
    )

    save_table(
        breakeven,
        "table_break_even"
    )

    print("\nSaved paper tables:")
    for path in sorted(RESULTS.glob("table_*.csv")):
        print(f"- {path}")
    for path in sorted(RESULTS.glob("table_*.tex")):
        print(f"- {path}")


if __name__ == "__main__":
    main()