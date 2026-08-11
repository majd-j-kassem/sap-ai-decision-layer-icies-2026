# scripts_and_models/19_cost_sensitive_analysis.py

import os
import pandas as pd
import numpy as np

DATA_PATH = "data/DataCoSupplyChainDataset.csv"
RESULT_PATH = "results/cost_sensitive_analysis.csv"

TARGET = "Late_delivery_risk"

# Adjustable business-cost assumptions
COST_LATE_DELIVERY = 100.0
COST_EXPEDITION = 25.0
COST_PRIORITY_REVIEW = 10.0
COST_SHIPPING_REVIEW = 5.0


def classify_risk(probability):
    if probability >= 0.70:
        return "HIGH"
    elif probability >= 0.30:
        return "MEDIUM"
    return "LOW"


def decision_cost(risk, actual_late):
    if risk == "LOW":
        return COST_LATE_DELIVERY if actual_late == 1 else 0.0

    if risk == "MEDIUM":
        review_cost = COST_SHIPPING_REVIEW
        late_cost = COST_LATE_DELIVERY if actual_late == 1 else 0.0
        return review_cost + late_cost

    if risk == "HIGH":
        action_cost = COST_EXPEDITION + COST_PRIORITY_REVIEW
        late_cost = COST_LATE_DELIVERY if actual_late == 1 else 0.0
        return action_cost + late_cost

    raise ValueError(risk)


def baseline_cost(actual_late):
    return COST_LATE_DELIVERY if actual_late == 1 else 0.0


def main():
    print("=== Cost-Sensitive Decision Analysis ===")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH, encoding="latin1")

    required = [
        TARGET,
        "Shipping Mode",
        "Market",
        "Order Region",
        "Order Item Quantity",
        "Product Price",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Reproduce the predictive-risk logic using the observed
    # risk structure established in the previous experiments.
    shipping_risk = {
        "Same Day": 0.13,
        "Standard Class": 0.38,
        "Second Class": 0.75,
        "First Class": 0.96,
    }

    df["risk_probability"] = (
        df["Shipping Mode"]
        .map(shipping_risk)
        .fillna(df[TARGET].mean())
    )

    df["risk_level"] = df["risk_probability"].apply(
        classify_risk
    )

    df["actual_late"] = df[TARGET].astype(int)

    df["baseline_cost"] = df["actual_late"].apply(
        baseline_cost
    )

    df["decision_cost"] = [
        decision_cost(risk, late)
        for risk, late in zip(
            df["risk_level"],
            df["actual_late"],
        )
    ]

    df["cost_saving"] = (
        df["baseline_cost"] -
        df["decision_cost"]
    )

    summary = (
        df.groupby("risk_level")
        .agg(
            orders=("actual_late", "size"),
            late_orders=("actual_late", "sum"),
            mean_risk=("risk_probability", "mean"),
            baseline_cost=("baseline_cost", "sum"),
            decision_cost=("decision_cost", "sum"),
            cost_saving=("cost_saving", "sum"),
        )
        .reset_index()
    )

    summary["saving_per_order"] = (
        summary["cost_saving"] /
        summary["orders"]
    )

    baseline_total = df["baseline_cost"].sum()
    decision_total = df["decision_cost"].sum()

    total_saving = baseline_total - decision_total

    saving_percentage = (
        total_saving / baseline_total
        if baseline_total > 0
        else 0.0
    )

    print("\n=== Cost Summary ===")
    print(f"Baseline cost : {baseline_total:,.2f}")
    print(f"Decision cost : {decision_total:,.2f}")
    print(f"Cost saving   : {total_saving:,.2f}")
    print(
        f"Saving rate   : "
        f"{saving_percentage * 100:.2f}%"
    )

    print("\n=== Risk-Level Cost Analysis ===")
    print(summary.to_string(index=False))

    os.makedirs("results", exist_ok=True)

    summary.to_csv(
        RESULT_PATH,
        index=False,
    )

    overall = pd.DataFrame(
        [{
            "baseline_cost": baseline_total,
            "decision_cost": decision_total,
            "cost_saving": total_saving,
            "saving_rate": saving_percentage,
            "cost_late_delivery": COST_LATE_DELIVERY,
            "cost_expedition": COST_EXPEDITION,
            "cost_priority_review": COST_PRIORITY_REVIEW,
            "cost_shipping_review": COST_SHIPPING_REVIEW,
        }]
    )

    overall.to_csv(
        "results/cost_sensitive_overall.csv",
        index=False,
    )

    print("\nSaved:")
    print(f"- {RESULT_PATH}")
    print("- results/cost_sensitive_overall.csv")


if __name__ == "__main__":
    main()