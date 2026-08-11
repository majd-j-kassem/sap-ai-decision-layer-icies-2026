import os
import pandas as pd
import numpy as np
import joblib

DATA_PATH = "data/DataCoSupplyChainDataset.csv"
MODEL_PATH = "models/lightgbm_model.joblib"
OUTPUT_PATH = "results/economic_scenario_analysis.csv"

FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Item Quantity",
    "Product Price",
]

LATE_COSTS = [50, 100, 150, 200, 300, 500, 1000]
INTERVENTION_COSTS = [0, 5, 10, 20, 30, 50, 100]


def classify_risk(p):
    if p >= 0.70:
        return "HIGH"
    elif p >= 0.30:
        return "MEDIUM"
    return "LOW"


def main():
    print("=== Economic Scenario Analysis ===")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    df = pd.read_csv(DATA_PATH, encoding="latin1")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(df[FEATURES])[:, 1]

    df["probability"] = probabilities
    df["risk_level"] = df["probability"].apply(classify_risk)

    results = []

    print(f"Observations: {len(df):,}")

    for late_cost in LATE_COSTS:

        baseline_cost = (
            df["probability"] * late_cost
        ).sum()

        for intervention_cost in INTERVENTION_COSTS:

            decision_cost = np.zeros(len(df))

            low = df["risk_level"] == "LOW"
            medium = df["risk_level"] == "MEDIUM"
            high = df["risk_level"] == "HIGH"

            # LOW: no intervention.
            decision_cost[low] = (
                df.loc[low, "probability"] * late_cost
            )

            # MEDIUM: review cost is added without
            # assuming a risk reduction.
            decision_cost[medium] = (
                df.loc[medium, "probability"] * late_cost
                + intervention_cost
            )

            # HIGH: intervention reduces expected
            # residual risk by 40%.
            decision_cost[high] = (
                df.loc[high, "probability"]
                * 0.60
                * late_cost
                + intervention_cost
            )

            total_decision_cost = decision_cost.sum()

            saving = (
                baseline_cost
                - total_decision_cost
            )

            saving_rate = (
                saving / baseline_cost
                if baseline_cost > 0
                else 0
            )

            results.append({
                "late_delivery_cost": late_cost,
                "intervention_cost": intervention_cost,
                "baseline_cost": baseline_cost,
                "decision_cost": total_decision_cost,
                "cost_saving": saving,
                "saving_rate": saving_rate,
                "economically_beneficial": saving > 0,
            })

    results_df = pd.DataFrame(results)

    print("\n=== Scenario Results ===")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}"
        )
    )

    print("\n=== Economic Benefit ===")

    beneficial = results_df[
        results_df["economically_beneficial"]
    ]

    print(
        f"Beneficial scenarios: {len(beneficial)}"
    )

    if not beneficial.empty:

        best = beneficial.loc[
            beneficial["cost_saving"].idxmax()
        ]

        print("\nBest scenario:")
        print(best.to_string())

    print("\n=== Break-Even Intervention Cost ===")

    break_even = []

    for late_cost in LATE_COSTS:

        base = results_df[
            results_df["late_delivery_cost"] == late_cost
        ]

        positive = base[
            base["cost_saving"] >= 0
        ]

        if positive.empty:
            maximum = np.nan
        else:
            maximum = positive[
                "intervention_cost"
            ].max()

        break_even.append({
            "late_delivery_cost": late_cost,
            "maximum_profitable_intervention_cost":
                maximum,
        })

    break_even_df = pd.DataFrame(break_even)

    print(
        break_even_df.to_string(index=False)
    )

    os.makedirs("results", exist_ok=True)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    break_even_df.to_csv(
        "results/economic_break_even.csv",
        index=False
    )

    print("\nSaved:")
    print(f"- {OUTPUT_PATH}")
    print("- results/economic_break_even.csv")


if __name__ == "__main__":
    main()