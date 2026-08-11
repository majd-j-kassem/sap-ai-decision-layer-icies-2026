import os
import pandas as pd
import numpy as np
import joblib

DATA_PATH = "data/DataCoSupplyChainDataset.csv"
MODEL_PATH = "models/lightgbm_model.joblib"
OUTPUT_PATH = "results/end_to_end_economic_evaluation.csv"

FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Item Quantity",
    "Product Price",
]

TARGET = "Late_delivery_risk"

LATE_COST = 100.0
INTERVENTION_COST = 10.0


def classify_risk(probability):
    if probability >= 0.70:
        return "HIGH"
    elif probability >= 0.30:
        return "MEDIUM"
    return "LOW"


def calculate_decision_cost(probability, risk):
    baseline = probability * LATE_COST

    if risk == "LOW":
        return baseline

    if risk == "MEDIUM":
        return baseline + INTERVENTION_COST

    # Assumed intervention effectiveness for high-risk orders.
    residual_risk = probability * 0.60
    return residual_risk * LATE_COST + INTERVENTION_COST


def main():
    print("=== End-to-End Economic Evaluation ===")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    df = pd.read_csv(DATA_PATH, encoding="latin1")

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    model = joblib.load(MODEL_PATH)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"Observations: {len(df):,}")

    probabilities = model.predict_proba(X)[:, 1]

    df["predicted_probability"] = probabilities
    df["risk_level"] = [
        classify_risk(p)
        for p in probabilities
    ]

    df["baseline_cost"] = (
        df["predicted_probability"] * LATE_COST
    )

    df["decision_cost"] = [
        calculate_decision_cost(p, r)
        for p, r in zip(
            df["predicted_probability"],
            df["risk_level"],
        )
    ]

    df["cost_saving"] = (
        df["baseline_cost"] - df["decision_cost"]
    )

    df["sap_action"] = df["risk_level"].map({
        "LOW": "SAP_STANDARD_PROCESS",
        "MEDIUM": "SAP_SHIPPING_PLAN_REVIEW",
        "HIGH": "SAP_ORDER_PRIORITY_REVIEW",
    })

    df["human_review"] = df["risk_level"] != "LOW"

    summary = (
        df.groupby("risk_level")
        .agg(
            orders=("risk_level", "size"),
            mean_probability=("predicted_probability", "mean"),
            baseline_cost=("baseline_cost", "sum"),
            decision_cost=("decision_cost", "sum"),
            cost_saving=("cost_saving", "sum"),
            human_review_orders=("human_review", "sum"),
        )
        .reset_index()
    )

    summary["saving_rate"] = (
        summary["cost_saving"]
        / summary["baseline_cost"]
    )

    print("\n=== Economic Results by Risk Level ===")
    print(summary.to_string(index=False))

    baseline_total = df["baseline_cost"].sum()
    decision_total = df["decision_cost"].sum()
    saving_total = df["cost_saving"].sum()

    saving_rate = saving_total / baseline_total

    print("\n=== Overall Economic Impact ===")
    print(f"Baseline Cost : ${baseline_total:,.2f}")
    print(f"Decision Cost : ${decision_total:,.2f}")
    print(f"Cost Saving   : ${saving_total:,.2f}")
    print(f"Saving Rate   : {saving_rate:.2%}")

    print("\n=== Economic Interpretation ===")

    if saving_total > 0:
        print(
            "The simulated decision layer produces a "
            "positive economic benefit under the defined "
            "cost and intervention assumptions."
        )
    else:
        print(
            "The simulated decision layer does not produce "
            "a positive economic benefit under the defined "
            "cost and intervention assumptions."
        )

    os.makedirs("results", exist_ok=True)

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()