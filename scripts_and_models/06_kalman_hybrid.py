import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/DataCoSupplyChainDataset.csv"
RESULTS_PATH = "results/kalman_hybrid_results.csv"

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

    return df[required].copy()


def kalman_filter(signal, process_variance=0.01, measurement_variance=0.1):
    """
    Simple scalar Kalman filter.

    This is used only as a temporal smoothing mechanism.
    It does not use the target variable.
    """

    signal = np.asarray(signal, dtype=float)

    estimate = np.zeros(len(signal))

    estimate[0] = signal[0]

    error_covariance = 1.0

    for i in range(1, len(signal)):
        prediction = estimate[i - 1]
        prediction_error = (
            error_covariance
            + process_variance
        )

        kalman_gain = (
            prediction_error
            / (
                prediction_error
                + measurement_variance
            )
        )

        estimate[i] = (
            prediction
            + kalman_gain
            * (signal[i] - prediction)
        )

        error_covariance = (
            (1 - kalman_gain)
            * prediction_error
        )

    return estimate


def create_operational_signal(df):
    """
    Construct a target-independent operational signal.

    The signal is based only on predictors available
    before prediction.

    Higher values indicate a combination of:
    - more expensive orders
    - larger quantities
    """

    quantity = df["Order Item Quantity"].astype(float)
    price = df["Product Price"].astype(float)

    quantity_norm = (
        quantity - quantity.mean()
    ) / quantity.std()

    price_norm = (
        price - price.mean()
    ) / price.std()

    signal = (
        0.5 * quantity_norm
        + 0.5 * price_norm
    )

    return signal


def build_pipeline():
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
            (
                "scaler",
                StandardScaler(),
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

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
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


def evaluate_model(
    name,
    X_train,
    X_test,
    y_train,
    y_test,
):
    model = build_pipeline()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    result = {
        "model": name,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    print(f"\n=== {name} ===")

    print(
        f"Accuracy : {result['accuracy']:.4f}"
    )

    print(
        f"Precision: {result['precision']:.4f}"
    )

    print(
        f"Recall   : {result['recall']:.4f}"
    )

    print(
        f"F1       : {result['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {result['roc_auc']:.4f}"
    )

    return result


def main():

    print("=== Kalman Hybrid Analysis ===")

    df = load_data()

    print(
        f"Dataset shape: {df.shape}"
    )

    # -------------------------------------------------
    # Create predictor-only operational signal
    # -------------------------------------------------

    signal = create_operational_signal(df)

    # Apply Kalman filtering to the signal.
    filtered_signal = kalman_filter(
        signal.values
    )

    df["Kalman_Signal"] = filtered_signal

    # -------------------------------------------------
    # Train/test split
    # -------------------------------------------------

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET],
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    # -------------------------------------------------
    # Baseline model
    # -------------------------------------------------

    baseline_train = train_df[
        FEATURES
    ]

    baseline_test = test_df[
        FEATURES
    ]

    baseline_result = evaluate_model(
        "Baseline",
        baseline_train,
        baseline_test,
        y_train,
        y_test,
    )

    # -------------------------------------------------
    # Hybrid model
    # -------------------------------------------------

    hybrid_features = FEATURES + [
        "Kalman_Signal"
    ]

    hybrid_train = train_df[
        hybrid_features
    ]

    hybrid_test = test_df[
        hybrid_features
    ]

    # Build a separate pipeline for hybrid features.
    hybrid_categorical = CATEGORICAL_FEATURES

    hybrid_numeric = (
        NUMERIC_FEATURES
        + ["Kalman_Signal"]
    )

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
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    hybrid_preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                hybrid_categorical,
            ),
            (
                "numeric",
                numeric_pipeline,
                hybrid_numeric,
            ),
        ]
    )

    hybrid_model = Pipeline(
        steps=[
            (
                "preprocessor",
                hybrid_preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    hybrid_model.fit(
        hybrid_train,
        y_train,
    )

    hybrid_probability = (
        hybrid_model.predict_proba(
            hybrid_test
        )[:, 1]
    )

    hybrid_prediction = (
        hybrid_probability >= 0.5
    ).astype(int)

    hybrid_result = {
        "model": "Kalman Hybrid",
        "accuracy": accuracy_score(
            y_test,
            hybrid_prediction,
        ),
        "precision": precision_score(
            y_test,
            hybrid_prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            hybrid_prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            hybrid_prediction,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            hybrid_probability,
        ),
    }

    print("\n=== Kalman Hybrid ===")

    print(
        f"Accuracy : "
        f"{hybrid_result['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{hybrid_result['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{hybrid_result['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{hybrid_result['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{hybrid_result['roc_auc']:.4f}"
    )

    # -------------------------------------------------
    # Compare models
    # -------------------------------------------------

    results = pd.DataFrame(
        [
            baseline_result,
            hybrid_result,
        ]
    )

    results["delta_roc_auc"] = (
        results["roc_auc"]
        - baseline_result["roc_auc"]
    )

    results["delta_f1"] = (
        results["f1"]
        - baseline_result["f1"]
    )

    print("\n=== Hybrid Comparison ===")

    print(
        results.to_string(
            index=False
        )
    )

    os.makedirs(
        "results",
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        f"\nSaved: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()