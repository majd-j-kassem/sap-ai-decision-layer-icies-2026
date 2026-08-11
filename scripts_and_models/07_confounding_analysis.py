import os

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


DATA_PATH = "data/DataCoSupplyChainDataset.csv"
RESULTS_PATH = "results/confounding_analysis.csv"

TARGET = "Late_delivery_risk"

PRIMARY_FEATURE = "Shipping Mode"

CONTROL_CATEGORICAL = [
    "Market",
    "Order Region",
]

CONTROL_NUMERIC = [
    "Order Item Quantity",
    "Product Price",
]


def load_data():
    df = pd.read_csv(
        DATA_PATH,
        encoding="ISO-8859-1",
    )

    features = [
        PRIMARY_FEATURE,
        *CONTROL_CATEGORICAL,
        *CONTROL_NUMERIC,
    ]

    missing = [c for c in features + [TARGET] if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[features + [TARGET]].copy()


def chi_square_test(df, feature):
    table = pd.crosstab(
        df[feature],
        df[TARGET],
    )

    chi2, p, dof, expected = chi2_contingency(table)

    return {
        "feature": feature,
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
    }


def build_pipeline(features):
    categorical = [
        f for f in features
        if f in [PRIMARY_FEATURE, *CONTROL_CATEGORICAL]
    ]

    numeric = [
        f for f in features
        if f in CONTROL_NUMERIC
    ]

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
                categorical,
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric,
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
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(df, features, label):
    X = df[features]
    y = df[TARGET]

    pipeline = build_pipeline(features)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print(f"\n=== {label} ===")
    print(f"Features: {features}")
    print(
        f"CV ROC-AUC: "
        f"{scores.mean():.4f} +/- {scores.std():.4f}"
    )

    return {
        "model": label,
        "features": ", ".join(features),
        "n_features": len(features),
        "cv_roc_auc_mean": scores.mean(),
        "cv_roc_auc_std": scores.std(),
    }


def main():
    print("=== Confounding Analysis ===")

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    print("\n=== Univariate Association ===")

    association_results = []

    for feature in [
        PRIMARY_FEATURE,
        *CONTROL_CATEGORICAL,
    ]:
        result = chi_square_test(df, feature)

        association_results.append(result)

        print(
            f"{feature:20s} "
            f"chi2={result['chi2']:.4f} "
            f"dof={result['dof']} "
            f"p={result['p_value']:.6e}"
        )

    print("\n=== Incremental Control Analysis ===")

    feature_sets = [
        (
            [PRIMARY_FEATURE],
            "Shipping Mode only",
        ),
        (
            [
                PRIMARY_FEATURE,
                "Market",
            ],
            "Shipping Mode + Market",
        ),
        (
            [
                PRIMARY_FEATURE,
                "Market",
                "Order Region",
            ],
            "Shipping Mode + Market + Region",
        ),
        (
            [
                PRIMARY_FEATURE,
                "Market",
                "Order Region",
                "Order Item Quantity",
                "Product Price",
            ],
            "Full adjusted model",
        ),
    ]

    model_results = []

    for features, label in feature_sets:
        result = evaluate_model(
            df,
            features,
            label,
        )

        model_results.append(result)

    results = pd.DataFrame(model_results)

    os.makedirs("results", exist_ok=True)

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    association_df = pd.DataFrame(
        association_results
    )

    association_df.to_csv(
        "results/confounding_association.csv",
        index=False,
    )

    print("\n=== Confounding Analysis Summary ===")
    print(
        results.to_string(index=False)
    )

    print("\nSaved:")
    print("- results/confounding_analysis.csv")
    print("- results/confounding_association.csv")


if __name__ == "__main__":
    main()