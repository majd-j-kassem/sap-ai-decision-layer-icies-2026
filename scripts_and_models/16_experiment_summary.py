import os
import pandas as pd

RESULTS_DIR = "results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "experiment_summary.csv")


def load_result(filename, label):
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None

    df = pd.read_csv(path)
    df["source"] = label
    return df


def main():
    print("=== Experiment Summary ===")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    datasets = []

    files = [
        ("statistical_validation_summary.csv", "Statistical Validation"),
        ("xgboost_lightgbm_results.csv", "XGBoost / LightGBM"),
        ("robustness_summary.csv", "Robustness"),
        ("decision_layer_summary.csv", "Decision Layer"),
        ("human_in_the_loop_summary.csv", "Human-in-the-Loop"),
    ]

    for filename, label in files:
        df = load_result(filename, label)

        if df is not None:
            datasets.append(df)

    if not datasets:
        raise RuntimeError("No result files found.")

    print("\n=== Available Results ===")

    for df in datasets:
        print(f"\nSource: {df['source'].iloc[0]}")
        print(df.to_string(index=False))

    summary_rows = []

    # Statistical validation
    path = os.path.join(
        RESULTS_DIR,
        "statistical_validation_summary.csv",
    )

    if os.path.exists(path):
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            summary_rows.append(
                {
                    "analysis": "Cross-Validation",
                    "model": row.get("model"),
                    "metric": "ROC-AUC",
                    "mean": row.get("mean_roc_auc"),
                    "std": row.get("std_roc_auc"),
                    "min": row.get("min_roc_auc"),
                    "max": row.get("max_roc_auc"),
                }
            )

    # XGBoost / LightGBM
    path = os.path.join(
        RESULTS_DIR,
        "xgboost_lightgbm_results.csv",
    )

    if os.path.exists(path):
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            summary_rows.append(
                {
                    "analysis": "Gradient Boosting",
                    "model": row.get("model"),
                    "metric": "ROC-AUC",
                    "mean": row.get("cv_roc_auc_mean"),
                    "std": row.get("cv_roc_auc_std"),
                    "min": None,
                    "max": None,
                }
            )

            summary_rows.append(
                {
                    "analysis": "Gradient Boosting",
                    "model": row.get("model"),
                    "metric": "PR-AUC",
                    "mean": row.get("cv_pr_auc_mean"),
                    "std": row.get("cv_pr_auc_std"),
                    "min": None,
                    "max": None,
                }
            )

    # Robustness
    path = os.path.join(
        RESULTS_DIR,
        "robustness_summary.csv",
    )

    if os.path.exists(path):
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            summary_rows.append(
                {
                    "analysis": "Random Seed Robustness",
                    "model": row.get("model"),
                    "metric": "ROC-AUC",
                    "mean": row.get("seed_mean_auc"),
                    "std": row.get("seed_std_auc"),
                    "min": row.get("seed_min_auc"),
                    "max": row.get("seed_max_auc"),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== Consolidated Experimental Summary ===")
    print(summary_df.to_string(index=False))

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()