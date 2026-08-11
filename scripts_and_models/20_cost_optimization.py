# scripts_and_models/20_cost_optimization.py

import os
import joblib
import numpy as np
import pandas as pd

DATA_PATH = "data/DataCoSupplyChainDataset.csv"
MODEL_PATH = "models/lightgbm_model.joblib"

RESULT_PATH = "results/cost_optimization.csv"

TARGET = "Late_delivery_risk"

FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Item Quantity",
    "Product Price",
]

# Business-cost scenarios
SCENARIOS = {
    "BASE": {
        "late": 100.0,
        "shipping_review": 5.0,
        "expedition": 25.0,
        "priority_review": 10.0,
    },
    "HIGH_LATE_COST": {
        "late": 200.0,
        "shipping_review": 5.0,
        "expedition": 25.0,
        "priority_review": 10.0,
    },
    "LOW_INTERVENTION_COST": {
        "late": 100.0,
        "shipping_review": 2.0,
        "expedition": 10.0,
        "priority_review": 5.0,
    },
}


def decision_cost(
    probability,
    actual,
    low_threshold,
    high_threshold,
    costs,
):
    if probability < low_threshold:
        intervention_cost = 0.0

    elif probability < high_threshold:
        intervention_cost = costs["shipping_review"]

    else:
        intervention_cost = (
            costs["expedition"]
            + costs["priority_review"]
        )

    late_cost = (
        costs["late"]
        if actual == 1
        else 0.0
    )

    return intervention_cost + late_cost


def main():

    print("=== Cost-Sensitive Threshold Optimization ===")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin1",
    )

    missing = [
        column
        for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    model = joblib.load(MODEL_PATH)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print("\n=== Generating Predictions ===")

    probabilities = model.predict_proba(X)[:, 1]

    df["predicted_probability"] = probabilities
    df["actual_late"] = y

    print(
        f"Observations: {len(df):,}"
    )

    results = []

    # Threshold search
    low_thresholds = np.arange(
        0.10,
        0.70,
        0.05,
    )

    high_thresholds = np.arange(
        0.40,
        0.96,
        0.05,
    )

    for scenario_name, costs in SCENARIOS.items():

        baseline_cost = (
            y.sum() * costs["late"]
        )

        best = None

        for low_threshold in low_thresholds:

            for high_threshold in high_thresholds:

                if high_threshold <= low_threshold:
                    continue

                total_cost = 0.0

                for probability, actual in zip(
                    probabilities,
                    y,
                ):
                    total_cost += decision_cost(
                        probability,
                        actual,
                        low_threshold,
                        high_threshold,
                        costs,
                    )

                saving = (
                    baseline_cost -
                    total_cost
                )

                saving_rate = (
                    saving / baseline_cost
                    if baseline_cost > 0
                    else 0.0
                )

                candidate = {
                    "scenario": scenario_name,
                    "low_threshold": low_threshold,
                    "high_threshold": high_threshold,
                    "baseline_cost": baseline_cost,
                    "optimized_cost": total_cost,
                    "cost_saving": saving,
                    "saving_rate": saving_rate,
                }

                if (
                    best is None
                    or total_cost
                    < best["optimized_cost"]
                ):
                    best = candidate

        results.append(best)

    results_df = pd.DataFrame(results)

    print("\n=== Optimization Results ===")

    display_columns = [
        "scenario",
        "low_threshold",
        "high_threshold",
        "baseline_cost",
        "optimized_cost",
        "cost_saving",
        "saving_rate",
    ]

    print(
        results_df[
            display_columns
        ].to_string(index=False)
    )

    print("\n=== Interpretation ===")

    for _, row in results_df.iterrows():

        if row["cost_saving"] > 0:

            print(
                f'{row["scenario"]}: '
                f'potential saving = '
                f'{row["cost_saving"]:,.2f} '
                f'({row["saving_rate"] * 100:.2f}%)'
            )

        else:

            print(
                f'{row["scenario"]}: '
                f'NO ECONOMIC BENEFIT under '
                f'this cost structure.'
            )

    os.makedirs(
        "results",
        exist_ok=True,
    )

    results_df.to_csv(
        RESULT_PATH,
        index=False,
    )

    print("\nSaved:")
    print(
        f"- {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()