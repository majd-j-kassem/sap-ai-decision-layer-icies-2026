import os
import json
import uuid
from datetime import datetime

import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "sap_workflow_audit_trail.csv",
)

TEST_ORDERS = [
    {
        "shipping_mode": "Same Day",
        "market": "USCA",
        "order_region": "Canada",
        "order_item_quantity": 1,
        "product_price": 20,
        "case": "LOW",
    },
    {
        "shipping_mode": "Standard Class",
        "market": "USCA",
        "order_region": "West of USA",
        "order_item_quantity": 2,
        "product_price": 59.99,
        "case": "MEDIUM",
    },
    {
        "shipping_mode": "First Class",
        "market": "Europe",
        "order_region": "Western Europe",
        "order_item_quantity": 4,
        "product_price": 250,
        "case": "HIGH",
    },
]


def call_decision_api(order):
    payload = {
        "shipping_mode": order["shipping_mode"],
        "market": order["market"],
        "order_region": order["order_region"],
        "order_item_quantity": order["order_item_quantity"],
        "product_price": order["product_price"],
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def determine_workflow(decision):
    risk = decision["prediction"]["risk_level"]

    if risk == "LOW":
        return {
            "workflow_status": "AUTO_APPROVED",
            "human_action": "NOT_REQUIRED",
            "workflow_step": "AUTOMATIC_PROCESSING",
        }

    if risk == "MEDIUM":
        return {
            "workflow_status": "PENDING_HUMAN_REVIEW",
            "human_action": "REVIEW_REQUIRED",
            "workflow_step": "SHIPPING_PLAN_REVIEW",
        }

    return {
        "workflow_status": "PENDING_PRIORITY_REVIEW",
        "human_action": "REVIEW_REQUIRED",
        "workflow_step": "ORDER_PRIORITY_REVIEW",
    }


def simulate_human_decision(risk_level):
    if risk_level == "LOW":
        return {
            "human_decision": "NOT_REQUIRED",
            "final_status": "APPROVED",
        }

    return {
        "human_decision": "APPROVED",
        "final_status": "APPROVED",
    }


def create_audit_record(order, decision):
    risk = decision["prediction"]["risk_level"]
    workflow = determine_workflow(decision)
    human = simulate_human_decision(risk)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "workflow_id": str(uuid.uuid4()),
        "case": order["case"],
        "risk_level": risk,
        "late_delivery_probability": decision[
            "prediction"
        ]["late_delivery_probability"],
        "recommended_action": decision[
            "decision"
        ]["recommended_action"],
        "human_review_required": decision[
            "decision"
        ]["human_review_required"],
        "workflow_status": workflow[
            "workflow_status"
        ],
        "workflow_step": workflow[
            "workflow_step"
        ],
        "human_action": workflow[
            "human_action"
        ],
        "human_decision": human[
            "human_decision"
        ],
        "final_status": human[
            "final_status"
        ],
        "sap_status": "SIMULATED",
        "sap_target": decision[
            "sap_integration"
        ]["target"],
    }


def main():
    print("=== SAP Workflow Simulation ===")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = []

    for index, order in enumerate(TEST_ORDERS, start=1):
        print(f"\n=== Test Case {index}: {order['case']} ===")

        try:
            decision = call_decision_api(order)

            record = create_audit_record(
                order,
                decision,
            )

            records.append(record)

            print(
                f"Risk Level       : "
                f"{record['risk_level']}"
            )
            print(
                f"Probability      : "
                f"{record['late_delivery_probability']}"
            )
            print(
                f"SAP Action       : "
                f"{record['recommended_action']}"
            )
            print(
                f"Workflow Status  : "
                f"{record['workflow_status']}"
            )
            print(
                f"Human Decision   : "
                f"{record['human_decision']}"
            )
            print(
                f"Final Status     : "
                f"{record['final_status']}"
            )

        except requests.RequestException as exc:
            print(
                f"API request failed: {exc}"
            )

    if not records:
        raise RuntimeError(
            "No workflow records were generated."
        )

    df = pd.DataFrame(records)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n=== Workflow Summary ===")
    print(
        df[
            [
                "case",
                "risk_level",
                "workflow_status",
                "human_decision",
                "final_status",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()