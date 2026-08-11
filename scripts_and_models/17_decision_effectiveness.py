import os
import pandas as pd
from scipy.stats import chi2_contingency

DATA_PATH = "results/decision_layer_summary.csv"
OUTPUT_PATH = "results/decision_effectiveness.csv"


def main():
    print("=== Decision Effectiveness Analysis ===")

    df = pd.read_csv(DATA_PATH)

    required = [
        "risk_level",
        "orders",
        "observed_late_rate",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Aggregate risk-level statistics
    summary = (
        df.groupby("risk_level", as_index=False)
        .agg(
            orders=("orders", "sum"),
            mean_risk=("mean_risk", "mean"),
            observed_late_rate=("observed_late_rate", "mean"),
        )
    )

    risk_order = ["LOW", "MEDIUM", "HIGH"]

    summary["risk_level"] = pd.Categorical(
        summary["risk_level"],
        categories=risk_order,
        ordered=True,
    )

    summary = summary.sort_values("risk_level")

    print("\n=== Risk-Level Effectiveness ===")
    print(summary.to_string(index=False))

    # Convert rates to approximate event counts
    summary["late_orders"] = (
        summary["orders"] *
        summary["observed_late_rate"]
    ).round().astype(int)

    summary["on_time_orders"] = (
        summary["orders"] -
        summary["late_orders"]
    )

    print("\n=== Observed Counts ===")
    print(
        summary[
            [
                "risk_level",
                "orders",
                "late_orders",
                "on_time_orders",
            ]
        ].to_string(index=False)
    )

    # Build actual contingency table
    contingency = summary[
        [
            "late_orders",
            "on_time_orders",
        ]
    ].to_numpy()

    chi2, p_value, dof, expected = chi2_contingency(
        contingency
    )

    print("\n=== Chi-Square Test ===")
    print(f"Chi-square : {chi2:.4f}")
    print(f"dof        : {dof}")
    print(f"p-value    : {p_value:.6e}")
    print(f"Significant: {p_value < 0.05}")

    # Monotonicity
    rates = summary["observed_late_rate"].tolist()

    monotonic = all(
        rates[i] <= rates[i + 1]
        for i in range(len(rates) - 1)
    )

    print("\n=== Monotonic Risk Check ===")
    print(
        "Observed late-delivery rate increases with risk:",
        monotonic,
    )

    # Risk separation
    low_rate = summary.loc[
        summary["risk_level"] == "LOW",
        "observed_late_rate",
    ].iloc[0]

    high_rate = summary.loc[
        summary["risk_level"] == "HIGH",
        "observed_late_rate",
    ].iloc[0]

    risk_gap = high_rate - low_rate

    relative_increase = (
        risk_gap / low_rate
        if low_rate > 0
        else None
    )

    print("\n=== Risk Separation ===")
    print(f"LOW late rate : {low_rate:.4f}")
    print(f"HIGH late rate: {high_rate:.4f}")
    print(f"Absolute gap  : {risk_gap:.4f}")
    print(
        f"Relative increase: {relative_increase:.2%}"
    )

    summary["risk_gap_high_vs_low"] = risk_gap
    summary["relative_high_vs_low"] = relative_increase
    summary["monotonic_risk_order"] = monotonic
    summary["chi_square"] = chi2
    summary["chi_square_dof"] = dof
    summary["chi_square_p_value"] = p_value

    os.makedirs("results", exist_ok=True)

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved:")
    print(f"- {OUTPUT_PATH}")


if __name__ == "__main__":
    main()