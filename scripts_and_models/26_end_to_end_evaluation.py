import os
import pandas as pd

RESULTS_PATH = "results"

TEST_CASES = [
    {
        "case": "Test Case 1",
        "shipping_mode": "Same Day",
        "market": "USCA",
        "order_region": "Canada",
        "quantity": 1,
        "price": 20.0,
        "probability": 0.1309,
        "risk_level": "LOW",
        "sap_action": "SAP_STANDARD_PROCESS",
        "human_review": False,
        "workflow_status": "AUTO_APPROVED",
        "final_status": "APPROVED",
    },
    {
        "case": "Test Case 2",
        "shipping_mode": "Standard Class",
        "market": "USCA",
        "order_region": "West of USA",
        "quantity": 2,
        "price": 59.99,
        "probability": 0.3820,
        "risk_level": "MEDIUM",
        "sap_action": "SAP_SHIPPING_PLAN_REVIEW",
        "human_review": True,
        "workflow_status": "PENDING_HUMAN_REVIEW",
        "final_status": "APPROVED",
    },
    {
        "case": "Test Case 3",
        "shipping_mode": "First Class",
        "market": "Europe",
        "order_region": "Western Europe",
        "quantity": 4,
        "price": 250.0,
        "probability": 0.9556,
        "risk_level": "HIGH",
        "sap_action": "SAP_ORDER_PRIORITY_REVIEW",
        "human_review": True,
        "workflow_status": "PENDING_PRIORITY_REVIEW",
        "final_status": "APPROVED",
    },
]


def economic_impact(probability, late_cost=100.0, intervention_cost=10.0):
    baseline = probability * late_cost

    if probability < 0.30:
        decision_cost = baseline
    elif probability < 0.70:
        decision_cost = baseline + intervention_cost
    else:
        decision_cost = probability * late_cost * 0.60 + intervention_cost

    saving = baseline - decision_cost

    return baseline, decision_cost, saving


def main():
    print("=== End-to-End SAP Decision Evaluation ===")

    rows = []

    for case in TEST_CASES:
        baseline, decision_cost, saving = economic_impact(
            case["probability"]
        )

        row = {
            **case,
            "baseline_expected_cost": round(baseline, 4),
            "decision_cost": round(decision_cost, 4),
            "expected_saving": round(saving, 4),
        }

        rows.append(row)

        print(f"\n=== {case['case']} ===")
        print(f"Risk Level       : {case['risk_level']}")
        print(f"Probability      : {case['probability']:.4f}")
        print(f"SAP Action       : {case['sap_action']}")
        print(f"Human Review     : {case['human_review']}")
        print(f"Workflow Status  : {case['workflow_status']}")
        print(f"Final Status     : {case['final_status']}")
        print(f"Baseline Cost    : ${baseline:.2f}")
        print(f"Decision Cost    : ${decision_cost:.2f}")
        print(f"Expected Saving  : ${saving:.2f}")

    df = pd.DataFrame(rows)

    os.makedirs(RESULTS_PATH, exist_ok=True)

    output_path = os.path.join(
        RESULTS_PATH,
        "end_to_end_evaluation.csv"
    )

    df.to_csv(output_path, index=False)

    print("\n=== End-to-End Summary ===")

    print(
        df[
            [
                "case",
                "risk_level",
                "probability",
                "sap_action",
                "human_review",
                "workflow_status",
                "final_status",
                "expected_saving",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()