import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


DATA_PATH = "data/DataCoSupplyChainDataset.csv"
MODEL_PATH = "models/lightgbm_model.joblib"

OUTPUT_PATH = "results/calibration_analysis.csv"
SUMMARY_PATH = "results/calibration_summary.csv"
LATEX_PATH = "results/table_calibration.tex"


def expected_calibration_error(y_true, probabilities, n_bins=10):
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

        if mask.sum() == 0:
            continue

        confidence = probabilities[mask].mean()
        accuracy = y_true[mask].mean()

        ece += (
            mask.sum() / len(y_true)
        ) * abs(confidence - accuracy)

    return ece


def main():
    print("=== Calibration Analysis ===")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin1",
    )

    target = "Late_delivery_risk"

    features = [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Order Item Quantity",
        "Product Price",
    ]

    X = df[features]
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

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
    print(f"Brier Score : {brier:.6f}")
    print(f"ECE         : {ece:.6f}")

    print("\n=== Calibration Curve ===")

    rows = []

    for i, (predicted, observed) in enumerate(
        zip(mean_predicted, fraction_positive),
        start=1,
    ):
        calibration_error = abs(
            predicted - observed
        )

        print(
            f"Bin {i:02d} | "
            f"Predicted={predicted:.4f} | "
            f"Observed={observed:.4f} | "
            f"Error={calibration_error:.4f}"
        )

        rows.append(
            {
                "bin": i,
                "mean_predicted_probability": predicted,
                "observed_frequency": observed,
                "absolute_calibration_error": calibration_error,
            }
        )

    calibration_df = pd.DataFrame(rows)

    # Save CSV calibration data
    calibration_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # Save calibration summary
    summary = pd.DataFrame(
        [
            {
                "model": "LightGBM",
                "brier_score": brier,
                "expected_calibration_error": ece,
                "test_samples": len(y_test),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # Generate LaTeX table
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