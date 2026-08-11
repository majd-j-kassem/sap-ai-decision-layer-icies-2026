import os
import pandas as pd

RESULTS_DIR = "results"


def load_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        print(f"Warning: missing {filename}")
        return None

    return pd.read_csv(path)


def main():
    print("=== Experimental Results Summary ===")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    outputs = {}

    files = {
        "model_results": "model_results.csv",
        "diagnostics": "model_diagnostics.csv",
        "ablation": "feature_ablation.csv",
        "confounding": "confounding_analysis.csv",
        "statistical": "statistical_validation_summary.csv",
        "pairwise": "pairwise_model_tests.csv",
        "xgb_lgbm": "xgboost_lightgbm_results.csv",
        "interpretation": "shap_feature_groups.csv",
        "robustness": "robustness_summary.csv",
        "kalman": "kalman_hybrid_results.csv",
        "decision": "decision_layer_summary.csv",
        "sap": "sap_integration_results.csv",
        "human": "human_in_the_loop_summary.csv",
    }

    for key, filename in files.items():
        outputs[key] = load_csv(filename)

    # ---------------------------------------------------------
    # 1. Core predictive performance
    # ---------------------------------------------------------

    print("\n=== Core Predictive Performance ===")

    if outputs["statistical"] is not None:
        df = outputs["statistical"].copy()

        columns = [
            "model",
            "mean_roc_auc",
            "std_roc_auc",
            "min_roc_auc",
            "max_roc_auc",
        ]

        columns = [c for c in columns if c in df.columns]

        core = df[columns].sort_values(
            "mean_roc_auc",
            ascending=False,
        )

        print(core.to_string(index=False))

        core.to_csv(
            os.path.join(
                RESULTS_DIR,
                "paper_table_model_performance.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # 2. Statistical significance
    # ---------------------------------------------------------

    print("\n=== Pairwise Statistical Tests ===")

    if outputs["pairwise"] is not None:
        df = outputs["pairwise"].copy()

        print(df.to_string(index=False))

        df.to_csv(
            os.path.join(
                RESULTS_DIR,
                "paper_table_statistical_tests.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # 3. Feature ablation
    # ---------------------------------------------------------

    print("\n=== Feature Ablation ===")

    if outputs["ablation"] is not None:
        df = outputs["ablation"].copy()

        columns = [
            "feature_group",
            "model",
            "n_features",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]

        columns = [c for c in columns if c in df.columns]

        ablation = df[columns].sort_values(
            "roc_auc",
            ascending=False,
        )

        print(ablation.to_string(index=False))

        ablation.to_csv(
            os.path.join(
                RESULTS_DIR,
                "paper_table_feature_ablation.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # 4. XGBoost / LightGBM
    # ---------------------------------------------------------

    print("\n=== Advanced Models ===")

    if outputs["xgb_lgbm"] is not None:
        df = outputs["xgb_lgbm"].copy()

        columns = [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
            "cv_pr_auc_mean",
            "cv_pr_auc_std",
        ]

        columns = [c for c in columns if c in df.columns]

        advanced = df[columns]

        print(advanced.to_string(index=False))

        advanced.to_csv(
            os.path.join(
                RESULTS_DIR,
                "paper_table_advanced_models.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # 5. Robustness
    # ---------------------------------------------------------

    print("\n=== Robustness ===")

    if outputs["robustness"] is not None:
        df = outputs["robustness"].copy()

        print(df.to_string(index=False))

        df.to_csv(
            os.path.join(
                RESULTS_DIR,
                "paper_table_robustness.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # 6. Model interpretation
    # ---------------------------------------------------------

    print("\n=== SHAP Feature Groups ===")

    if outputs["interpretation"] is not None:
        df = outputs["interpretation"].copy()

        print(df.to_string(index=False))

        df.to_csv(
            os.path.join(
                RESULTS_DIR,
                "paper_table_model_interpretation.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # 7. Kalman hybrid
    # ---------------------------------------------------------

    print("\n=== Kalman Hybrid ===")

    if outputs["kalman"] is not None:
        df = outputs["kalman"].copy()

        print(df.to_string(index=False))

        df.to_csv(
            os.path.join(
                RESULTS_DIR,
                "paper_table_kalman_hybrid.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # 8. Decision layer
    # ---------------------------------------------------------

    print("\n=== Decision Layer ===")

    if outputs["decision"] is not None:
        df = outputs["decision"].copy()

        print(df.to_string(index=False))

        df.to_csv(
            os.path.join(
                RESULTS_DIR,
                "paper_table_decision_layer.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # 9. SAP integration
    # ---------------------------------------------------------

    print("\n=== SAP Integration ===")

    if outputs["sap"] is not None:
        df = outputs["sap"].copy()

        print(df.to_string(index=False))

        df.to_csv(
            os.path.join(
                RESULTS_DIR,
                "paper_table_sap_integration.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # 10. Human-in-the-loop
    # ---------------------------------------------------------

    print("\n=== Human-in-the-Loop ===")

    if outputs["human"] is not None:
        df = outputs["human"].copy()

        print(df.to_string(index=False))

        df.to_csv(
            os.path.join(
                RESULTS_DIR,
                "paper_table_human_in_the_loop.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # Final manifest
    # ---------------------------------------------------------

    generated = [
        f
        for f in os.listdir(RESULTS_DIR)
        if f.startswith("paper_table_") and f.endswith(".csv")
    ]

    print("\n=== Generated Paper Tables ===")

    for filename in sorted(generated):
        print(f"- {filename}")

    print("\nExperimental summary completed.")


if __name__ == "__main__":
    main()