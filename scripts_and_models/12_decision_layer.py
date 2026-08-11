import os

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
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
    df = pd.read_csv(
        DATA_PATH,
        encoding="ISO-8859-1",
    )

    required = FEATURES + [TARGET]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    return X, y


def build_preprocessor():
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
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
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    return ColumnTransformer(
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


def build_model():
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )


def classify_risk(probability):
    if probability >= 0.75:
        return "HIGH"
    elif probability >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def recommend_action(
    risk_level,
    shipping_mode,
    product_price,
):
    if risk_level == "HIGH":
        if product_price >= 200:
            return "EXPEDITE_AND_PRIORITY_REVIEW"

        return "EXPEDITE_SHIPMENT"

    if risk_level == "MEDIUM":
        if shipping_mode in [
            "Standard Class",
            "Second Class",
        ]:
            return "REVIEW_SHIPPING_PLAN"

        return "MONITOR_ORDER"

    return "NO_ACTION"


def map_to_sap_action(action):
    """
    Conceptual SAP-oriented action mapping.
    This does not call SAP APIs.
    """

    mapping = {
        "EXPEDITE_AND_PRIORITY_REVIEW": (
            "SAP_ORDER_PRIORITY_REVIEW"
        ),
        "EXPEDITE_SHIPMENT": (
            "SAP_SHIPMENT_EXPEDITION"
        ),
        "REVIEW_SHIPPING_PLAN": (
            "SAP_SHIPPING_PLAN_REVIEW"
        ),
        "MONITOR_ORDER": (
            "SAP_ORDER_MONITORING"
        ),
        "NO_ACTION": (
            "SAP_STANDARD_PROCESS"
        ),
    }

    return mapping[action]


def main():
    print("=== SAP-Oriented Predictive Decision Layer ===")

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    X, y = load_data()

    print(
        f"Dataset shape: {X.shape}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    preprocessor = build_preprocessor()
    model = build_model()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    print("\n=== Training Predictive Model ===")

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    print("\n=== Predictive Performance ===")

    print(
        f"Accuracy : "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, predictions):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_test, predictions):.4f}"
    )

    print(
        f"F1       : "
        f"{f1_score(y_test, predictions):.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    decision_df = X_test.copy()

    decision_df["actual_late_delivery"] = (
        y_test.values
    )

    decision_df["predicted_probability"] = (
        probabilities
    )

    decision_df["risk_level"] = (
        decision_df["predicted_probability"]
        .apply(classify_risk)
    )

    decision_df["recommended_action"] = (
        decision_df.apply(
            lambda row: recommend_action(
                row["risk_level"],
                row["Shipping Mode"],
                row["Product Price"],
            ),
            axis=1,
        )
    )

    decision_df["sap_action"] = (
        decision_df["recommended_action"]
        .apply(map_to_sap_action)
    )

    decision_df["decision_confidence"] = (
        np.abs(
            decision_df[
                "predicted_probability"
            ] - 0.50
        ) * 2
    )

    print("\n=== Decision Distribution ===")

    print(
        decision_df[
            "risk_level"
        ].value_counts()
    )

    print("\n=== Recommended Actions ===")

    print(
        decision_df[
            "recommended_action"
        ].value_counts()
    )

    print("\n=== SAP-Oriented Actions ===")

    print(
        decision_df[
            "sap_action"
        ].value_counts()
    )

    output_path = (
        f"{RESULTS_DIR}/"
        "decision_layer_results.csv"
    )

    decision_df.to_csv(
        output_path,
        index=False,
    )

    summary = (
        decision_df
        .groupby(
            [
                "risk_level",
                "recommended_action",
                "sap_action",
            ]
        )
        .agg(
            orders=("risk_level", "size"),
            mean_risk=(
                "predicted_probability",
                "mean",
            ),
            observed_late_rate=(
                "actual_late_delivery",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        f"{RESULTS_DIR}/"
        "decision_layer_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nSaved:")
    print(
        f"- {output_path}"
    )
    print(
        f"- {summary_path}"
    )


if __name__ == "__main__":
    main()