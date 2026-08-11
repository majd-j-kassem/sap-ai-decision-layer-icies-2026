import os
import joblib
import numpy as np
import pandas as pd

DATA_PATH = "data/DataCoSupplyChainDataset.csv"
MODEL_PATH = "models/lightgbm_model.joblib"
OUTPUT_PATH = "results/cost_sensitivity_analysis.csv"

FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Item Quantity",
    "Product Price",
]

TARGET = "Late_delivery_risk"


def decision_action(score):
    if score >= 0.70:
        return "HIGH"
    elif score >= 0.30:
        return "MEDIUM"
    return "LOW"


def main():
    print("=== Cost Sensitivity Analysis ===")

    df = pd.read_csv(DATA_PATH, encoding="latin1")

    X = df[FEATURES]
    y = df[TARGET]

    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X)[:, 1]

    results = []

    late_costs = [50, 100, 150, 200, 300, 500, 1000]
    intervention_costs = [0, 5, 10, 20, 30, 50, 100]

    for late_cost in late_costs:
        for intervention_cost in intervention_costs:

            baseline_cost = y.sum() * late_cost

            intervention = np.where(
                probabilities >= 0.30,
                intervention_cost,
                0,
            )

            residual_late = np.where(
                probabilities >= 0.70,
                probabilities * 0.50,
                probabilities,
            )

            optimized_cost = (
                residual_late.sum() * late_cost
                + intervention.sum()
            )

            saving = baseline_cost - optimized_cost

            saving_rate = (
                saving / baseline_cost
                if baseline_cost > 0
                else 0
            )

            results.append({
                "late_delivery_cost": late_cost,
                "intervention_cost": intervention_cost,
                "baseline_cost": baseline_cost,
                "decision_cost": optimized_cost,
                "cost_saving": saving,
                "saving_rate": saving_rate,
                "economically_beneficial": saving > 0,
            })

    result_df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Sensitivity Results ===")
    print(result_df.to_string(index=False))

    beneficial = result_df[
        result_df["economically_beneficial"]
    ]

    print("\n=== Economic Benefit ===")

    if beneficial.empty:
        print("No economically beneficial scenario found.")
    else:
        print(
            f"Beneficial scenarios: {len(beneficial)}"
        )

        best = beneficial.loc[
            beneficial["cost_saving"].idxmax()
        ]

        print("\nBest scenario:")
        print(best.to_string())

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()