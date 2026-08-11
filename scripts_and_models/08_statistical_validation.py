import os

import numpy as np
import pandas as pd

from scipy.stats import ttest_rel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


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
        encoding="ISO-8859-1"
    )

    required = FEATURES + [TARGET]

    missing = [
        col for col in required
        if col not in df.columns
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
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
        ]
    )


def build_models():

    return {
        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),

        "CART":
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
    }


def evaluate_cv(X, y):

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    models = build_models()

    scores = {
        name: []
        for name in models
    }

    print("\n=== Repeated Cross-Validation ===")

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        for name, estimator in models.items():

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        build_preprocessor()
                    ),
                    (
                        "model",
                        estimator
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train
            )

            probabilities = (
                pipeline.predict_proba(X_test)[:, 1]
            )

            auc = roc_auc_score(
                y_test,
                probabilities
            )

            scores[name].append(auc)

            print(
                f"Fold {fold:02d} | "
                f"{name:20s} | "
                f"ROC-AUC={auc:.6f}"
            )

    return pd.DataFrame(scores)


def summarize_scores(scores):

    rows = []

    for model in scores.columns:

        values = scores[model]

        rows.append(
            {
                "model": model,
                "mean_roc_auc": values.mean(),
                "std_roc_auc": values.std(
                    ddof=1
                ),
                "min_roc_auc": values.min(),
                "max_roc_auc": values.max(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "mean_roc_auc",
        ascending=False
    )


def pairwise_tests(scores):

    models = list(scores.columns)

    results = []

    print(
        "\n=== Pairwise Statistical Tests ==="
    )

    for i in range(len(models)):

        for j in range(i + 1, len(models)):

            model_a = models[i]
            model_b = models[j]

            a = scores[model_a].values
            b = scores[model_b].values

            statistic, p_value = ttest_rel(
                a,
                b
            )

            mean_difference = (
                a.mean() - b.mean()
            )

            results.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_auc_a": a.mean(),
                    "mean_auc_b": b.mean(),
                    "mean_difference": mean_difference,
                    "t_statistic": statistic,
                    "p_value": p_value,
                    "significant_at_0_05": (
                        p_value < 0.05
                    ),
                }
            )

            print(
                f"{model_a} vs {model_b}: "
                f"ΔAUC={mean_difference:.6f}, "
                f"p={p_value:.6e}"
            )

    return pd.DataFrame(results)


def main():

    print(
        "=== Statistical Validation of Predictive Models ==="
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    X, y = load_data()

    print(
        f"Dataset shape: {X.shape}"
    )

    scores = evaluate_cv(
        X,
        y
    )

    summary = summarize_scores(
        scores
    )

    tests = pairwise_tests(
        scores
    )

    print(
        "\n=== Model Performance Summary ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n=== Statistical Test Results ==="
    )

    print(
        tests.to_string(
            index=False
        )
    )

    scores.to_csv(
        f"{RESULTS_DIR}/statistical_cv_scores.csv",
        index=False
    )

    summary.to_csv(
        f"{RESULTS_DIR}/statistical_validation_summary.csv",
        index=False
    )

    tests.to_csv(
        f"{RESULTS_DIR}/pairwise_model_tests.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "- results/statistical_cv_scores.csv"
    )
    print(
        "- results/statistical_validation_summary.csv"
    )
    print(
        "- results/pairwise_model_tests.csv"
    )


if __name__ == "__main__":
    main()