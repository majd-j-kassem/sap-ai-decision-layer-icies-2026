
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier


DATA_PATH = "data/DataCoSupplyChainDataset.csv"

OUTPUT_PATH = "results/calibration_analysis.csv"
SUMMARY_PATH = "results/calibration_summary.csv"
LATEX_PATH = "results/table_calibration.tex"


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
TARGET = "Late_delivery_risk"


def expected_calibration_error(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (probabilities >= bins[i])
                & (probabilities <= bins[i + 1])
            )
        else:
            mask = (
                (probabilities >= bins[i])
                & (probabilities < bins[i + 1])
            )

        if not mask.any():
            continue

        confidence = probabilities[mask].mean()
        observed_rate = y_true[mask].mean()

        ece += (
            mask.sum() / len(y_true)
        ) * abs(confidence - observed_rate)

    return ece


def build_preprocessor():
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


def build_lightgbm():
    return LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def main():
    print("=== Independent Holdout Calibration Analysis ===")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin1",
    )

    missing = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # Same holdout split as the training script
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Train only on the 80% training partition
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_lightgbm()),
        ]
    )

    model.fit(X_train, y_train)

    # Evaluate on the untouched 20% holdout partition
    probabilities = model.predict_proba(X_test)[:, 1]

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    ece = expected_calibration_error(
        y_test.to_numpy(),
        probabilities,
        n_bins=10,
    )

    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )

    print("\n=== Calibration Metrics ===")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples    : {len(y_test)}")
    print(f"Brier Score     : {brier:.6f}")
    print(f"ECE (10 bins)   : {ece:.6f}")

    rows = []

    print("\n=== Calibration Curve ===")

    for i, (predicted, observed) in enumerate(
        zip(mean_predicted, fraction_positive),
        start=1,
    ):
        error = abs(predicted - observed)

        print(
            f"Bin {i:02d} | "
            f"Predicted={predicted:.4f} | "
            f"Observed={observed:.4f} | "
            f"Error={error:.4f}"
        )

        rows.append(
            {
                "bin": i,
                "mean_predicted_probability": predicted,
                "observed_frequency": observed,
                "absolute_calibration_error": error,
            }
        )

    calibration_df = pd.DataFrame(rows)

    calibration_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "model": "LightGBM",
                "brier_score": brier,
                "expected_calibration_error": ece,
                "test_samples": len(y_test),
                "ece_bins": 10,
                "evaluation": "holdout",
            }
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    latex_df = calibration_df.rename(
        columns={
            "bin": "Bin",
            "mean_predicted_probability":
                "Mean Predicted Probability",
            "observed_frequency":
                "Observed Frequency",
            "absolute_calibration_error":
                "Absolute Calibration Error",
        }
    )

    latex_table = latex_df.to_latex(
        index=False,
        escape=True,
        float_format="%.4f",
        column_format="rrrr",
    )

    with open(
        LATEX_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(latex_table)

    print("\nSaved:")
    print(f"- {OUTPUT_PATH}")
    print(f"- {SUMMARY_PATH}")
    print(f"- {LATEX_PATH}")


if __name__ == "__main__":
    main()
