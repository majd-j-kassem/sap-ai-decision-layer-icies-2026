import json
import os
from datetime import datetime, timezone

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/DataCoSupplyChainDataset.csv"
RESULTS_DIR = "results"

TARGET = "Late_delivery_risk"

CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
]

NUMERIC_FEATURES = [
    "Order Item Quantity",
    "Product Price",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

    required = FEATURES + [
        TARGET,
        "Order Id",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def build_model():
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def classify_risk(probability):
    if probability >= 0.75:
        return "HIGH"

    if probability >= 0.50:
        return "MEDIUM"

    return "LOW"


def determine_action(risk_level):
    if risk_level == "HIGH":
        return "EXPEDITE_AND_PRIORITY_REVIEW"

    if risk_level == "MEDIUM":
        return "REVIEW_SHIPPING_PLAN"

    return "NO_ACTION"


def map_to_sap_action(action):
    mapping = {
        "NO_ACTION": "SAP_STANDARD_PROCESS",
        "REVIEW_SHIPPING_PLAN": "SAP_SHIPPING_PLAN_REVIEW",
        "EXPEDITE_SHIPMENT": "SAP_SHIPMENT_EXPEDITION",
        "EXPEDITE_AND_PRIORITY_REVIEW":
            "SAP_ORDER_PRIORITY_REVIEW",
        "MONITOR_ORDER": "SAP_ORDER_MONITORING",
    }

    return mapping[action]


def create_sap_request(row, probability, risk, action):
    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "timestamp": timestamp,
        "system": "SAP_ERP",
        "interface": "PREDICTIVE_DECISION_LAYER",
        "event_type": "DELIVERY_RISK_DECISION",
        "order_id": str(row["Order Id"]),
        "predicted_risk_probability": round(
            float(probability),
            6,
        ),
        "risk_level": risk,
        "decision": action,
        "sap_action": map_to_sap_action(action),
        "status": "SIMULATED",
    }


def simulate_sap_response(request):
    return {
        "order_id": request["order_id"],
        "sap_action": request["sap_action"],
        "status": "ACCEPTED",
        "execution_mode": "SIMULATION",
        "message": "SAP action accepted by prototype interface.",
    }


def main():
    print("=== SAP Integration Prototype ===")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print("\n=== Training Predictive Model ===")

    model = build_model()
    model.fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]

    records = []
    audit_records = []

    sample = df.copy()
    sample["risk_probability"] = probabilities

    for _, row in sample.iterrows():

        probability = row["risk_probability"]

        risk = classify_risk(probability)

        action = determine_action(risk)

        sap_request = create_sap_request(
            row,
            probability,
            risk,
            action,
        )

        sap_response = simulate_sap_response(
            sap_request
        )

        records.append(
            {
                "order_id": row["Order Id"],
                "risk_probability": probability,
                "risk_level": risk,
                "decision": action,
                "sap_action": sap_request["sap_action"],
                "sap_status": sap_response["status"],
            }
        )

        audit_records.append(
            {
                "timestamp": sap_request["timestamp"],
                "order_id": row["Order Id"],
                "prediction": round(
                    float(probability),
                    6,
                ),
                "risk_level": risk,
                "decision": action,
                "sap_action": sap_request["sap_action"],
                "execution_mode": "SIMULATION",
                "status": sap_response["status"],
            }
        )

    results = pd.DataFrame(records)

    audit = pd.DataFrame(audit_records)

    print("\n=== Decision Distribution ===")
    print(
        results["risk_level"]
        .value_counts()
    )

    print("\n=== SAP Action Distribution ===")
    print(
        results["sap_action"]
        .value_counts()
    )

    print("\n=== Integration Status ===")
    print(
        results["sap_status"]
        .value_counts()
    )

    results.to_csv(
        f"{RESULTS_DIR}/sap_integration_results.csv",
        index=False,
    )

    audit.to_csv(
        f"{RESULTS_DIR}/sap_audit_trail.csv",
        index=False,
    )

    sample_request = create_sap_request(
        df.iloc[0],
        probabilities[0],
        classify_risk(probabilities[0]),
        determine_action(
            classify_risk(probabilities[0])
        ),
    )

    with open(
        f"{RESULTS_DIR}/sample_sap_request.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            sample_request,
            file,
            indent=2,
        )

    print("\nSaved:")
    print(
        "- results/sap_integration_results.csv"
    )
    print(
        "- results/sap_audit_trail.csv"
    )
    print(
        "- results/sample_sap_request.json"
    )


if __name__ == "__main__":
    main()