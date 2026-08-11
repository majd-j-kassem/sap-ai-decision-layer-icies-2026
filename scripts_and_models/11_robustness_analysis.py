import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    return X, y


def build_preprocessor():
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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


def build_models(seed):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
        "CART": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def evaluate_seed_stability(X, y):
    print("\n=== Random Seed Stability ===")

    seeds = [7, 21, 42, 84, 123]
    rows = []

    for seed in seeds:
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        for name, estimator in build_models(seed).items():

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("model", estimator),
                ]
            )

            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )

            rows.append(
                {
                    "experiment": "random_seed",
                    "seed": seed,
                    "model": name,
                    "mean_roc_auc": scores.mean(),
                    "std_roc_auc": scores.std(),
                    "min_roc_auc": scores.min(),
                    "max_roc_auc": scores.max(),
                }
            )

            print(
                f"Seed {seed:3d} | "
                f"{name:20s} | "
                f"ROC-AUC={scores.mean():.6f} "
                f"+/- {scores.std():.6f}"
            )

    return pd.DataFrame(rows)


def evaluate_sample_stability(X, y):
    print("\n=== Sample Size Stability ===")

    fractions = [0.50, 0.70, 0.85, 1.00]
    seed = 42

    rows = []

    for fraction in fractions:

        if fraction < 1.0:
            from sklearn.model_selection import train_test_split

            X_sample, _, y_sample, _ = train_test_split(
                X,
                y,
                train_size=fraction,
                stratify=y,
                random_state=seed,
            )
        else:
            X_sample = X
            y_sample = y

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        for name, estimator in build_models(seed).items():

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("model", estimator),
                ]
            )

            scores = cross_val_score(
                pipeline,
                X_sample,
                y_sample,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )

            rows.append(
                {
                    "experiment": "sample_size",
                    "sample_fraction": fraction,
                    "sample_size": len(X_sample),
                    "model": name,
                    "mean_roc_auc": scores.mean(),
                    "std_roc_auc": scores.std(),
                }
            )

            print(
                f"Sample {fraction:.0%} "
                f"({len(X_sample):6d}) | "
                f"{name:20s} | "
                f"ROC-AUC={scores.mean():.6f}"
            )

    return pd.DataFrame(rows)


def evaluate_feature_stability(X, y):
    print("\n=== Feature Stability ===")

    feature_sets = {
        "Shipping Mode Only": [
            "Shipping Mode",
        ],
        "Shipping + Region": [
            "Shipping Mode",
            "Order Region",
        ],
        "All Categorical": [
            "Shipping Mode",
            "Market",
            "Order Region",
        ],
        "All Features": FEATURES,
    }

    seed = 42

    rows = []

    for group_name, feature_list in feature_sets.items():

        categorical = [
            f for f in feature_list
            if f in CATEGORICAL_FEATURES
        ]

        numeric = [
            f for f in feature_list
            if f in NUMERIC_FEATURES
        ]

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
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
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers = []

        if categorical:
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical,
                )
            )

        if numeric:
            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric,
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        X_subset = X[feature_list]

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        scores = cross_val_score(
            pipeline,
            X_subset,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        rows.append(
            {
                "experiment": "feature_set",
                "feature_group": group_name,
                "features": ", ".join(feature_list),
                "n_features": len(feature_list),
                "mean_roc_auc": scores.mean(),
                "std_roc_auc": scores.std(),
                "min_roc_auc": scores.min(),
                "max_roc_auc": scores.max(),
            }
        )

        print(
            f"{group_name:20s} | "
            f"ROC-AUC={scores.mean():.6f} "
            f"+/- {scores.std():.6f}"
        )

    return pd.DataFrame(rows)


def summarize_robustness(
    seed_results,
    sample_results,
    feature_results,
):
    print("\n=== Robustness Summary ===")

    summary = []

    for model in seed_results["model"].unique():

        subset = seed_results[
            seed_results["model"] == model
        ]

        summary.append(
            {
                "model": model,
                "seed_mean_auc": subset["mean_roc_auc"].mean(),
                "seed_std_auc": subset["mean_roc_auc"].std(),
                "seed_min_auc": subset["mean_roc_auc"].min(),
                "seed_max_auc": subset["mean_roc_auc"].max(),
            }
        )

    summary_df = pd.DataFrame(summary)

    print(summary_df.to_string(index=False))

    return summary_df


def main():

    print("=== Robustness Analysis ===")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y = load_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {FEATURES}")

    seed_results = evaluate_seed_stability(X, y)

    sample_results = evaluate_sample_stability(X, y)

    feature_results = evaluate_feature_stability(X, y)

    summary = summarize_robustness(
        seed_results,
        sample_results,
        feature_results,
    )

    seed_results.to_csv(
        f"{RESULTS_DIR}/robustness_seed_stability.csv",
        index=False,
    )

    sample_results.to_csv(
        f"{RESULTS_DIR}/robustness_sample_stability.csv",
        index=False,
    )

    feature_results.to_csv(
        f"{RESULTS_DIR}/robustness_feature_stability.csv",
        index=False,
    )

    summary.to_csv(
        f"{RESULTS_DIR}/robustness_summary.csv",
        index=False,
    )

    print("\nSaved:")
    print("- results/robustness_seed_stability.csv")
    print("- results/robustness_sample_stability.csv")
    print("- results/robustness_feature_stability.csv")
    print("- results/robustness_summary.csv")


if __name__ == "__main__":
    main()