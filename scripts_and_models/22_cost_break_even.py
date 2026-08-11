import os
import joblib
import numpy as np
import pandas as pd

DATA_PATH = "data/DataCoSupplyChainDataset.csv"
MODEL_PATH = "models/lightgbm_model.joblib"
OUTPUT_PATH = "results/cost_break_even.csv"

FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Item Quantity",
    "Product Price",
]

TARGET = "Late_delivery_risk"


def main():
    print("=== Cost Break-Even Analysis ===")

    df = pd.read_csv(DATA_PATH, encoding="latin1")
    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(df[FEATURES])[:, 1]
    actual = df[TARGET].to_numpy()

    intervention_mask = probabilities >= 0.30

    # Assume HIGH-risk intervention reduces late-delivery probability by 50%.
    residual_risk = np.where(
        probabilities >= 0.70,
        probabilities * 0.50,
        probabilities,
    )

    late_costs = [50, 100, 150, 200, 300, 500, 1000]

    rows = []

    intervention_orders = intervention_mask.sum()

    for late_cost in late_costs:

        baseline_cost = actual.sum() * late_cost

        residual_delivery_cost = (
            residual_risk.sum() * late_cost
        )

        avoided_cost = (
            baseline_cost - residual_delivery_cost
        )

        # Maximum intervention cost per intervention
        break_even_per_order = (
            avoided_cost / intervention_orders
            if intervention_orders > 0
            else 0
        )

        rows.append({
            "late_delivery_cost": late_cost,
            "intervention_orders": intervention_orders,
            "baseline_cost": baseline_cost,
            "avoided_late_delivery_cost": avoided_cost,
            "break_even_intervention_cost": break_even_per_order,
            "break_even_total_intervention_cost": avoided_cost,
        })

    result = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Break-Even Results ===")
    print(result.to_string(index=False))

    print("\n=== Interpretation ===")

    for _, row in result.iterrows():
        print(
            f"Late cost ${row['late_delivery_cost']:.0f}: "
            f"maximum intervention cost = "
            f"${row['break_even_intervention_cost']:.2f} per order"
        )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()