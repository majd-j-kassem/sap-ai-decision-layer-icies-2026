import os

import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier


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
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ]
    )


def get_feature_names(preprocessor):

    return preprocessor.get_feature_names_out()


def train_lightgbm(X, y):

    preprocessor = build_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    model = LGBMClassifier(
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

    model.fit(
        X_transformed,
        y
    )

    feature_names = get_feature_names(
        preprocessor
    )

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    return (
        preprocessor,
        model,
        X_transformed_df
    )


def extract_feature_importance(
    model,
    feature_names
):

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    importance["relative_importance"] = (
        importance["importance"]
        / importance["importance"].sum()
    )

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    return importance


def run_shap_analysis(
    model,
    X_transformed
):

    print(
        "\n=== SHAP Analysis ==="
    )

    # Use a representative sample to keep SHAP computation efficient.
    sample_size = min(
        10000,
        len(X_transformed)
    )

    X_sample = X_transformed.sample(
        n=sample_size,
        random_state=42
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    # SHAP versions may return either an array
    # or a list for binary classification.
    if isinstance(
        shap_values,
        list
    ):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(
        shap_values
    ).mean(
        axis=0
    )

    shap_importance = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "mean_abs_shap": mean_abs_shap,
        }
    )

    shap_importance[
        "relative_importance"
    ] = (
        shap_importance["mean_abs_shap"]
        / shap_importance[
            "mean_abs_shap"
        ].sum()
    )

    shap_importance = (
        shap_importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
    )

    return (
        shap_importance,
        shap_values,
        X_sample
    )


def aggregate_shipping_mode(
    shap_importance
):

    rows = []

    for feature_group in [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Order Item Quantity",
        "Product Price",
    ]:

        matching = shap_importance[
            shap_importance["feature"]
            .str.contains(
                feature_group,
                regex=False
            )
        ]

        if len(matching) == 0:
            continue

        rows.append(
            {
                "feature_group": feature_group,
                "mean_abs_shap": (
                    matching[
                        "mean_abs_shap"
                    ].sum()
                ),
            }
        )

    result = pd.DataFrame(rows)

    if len(result) > 0:
        result[
            "relative_importance"
        ] = (
            result["mean_abs_shap"]
            / result["mean_abs_shap"].sum()
        )

        result = result.sort_values(
            "mean_abs_shap",
            ascending=False
        )

    return result


def main():

    print(
        "=== Model Interpretation ==="
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    X, y = load_data()

    print(
        f"Dataset shape: {X.shape}"
    )

    (
        preprocessor,
        model,
        X_transformed
    ) = train_lightgbm(
        X,
        y
    )

    feature_names = (
        X_transformed.columns
    )

    print(
        "\n=== LightGBM Feature Importance ==="
    )

    importance = (
        extract_feature_importance(
            model,
            feature_names
        )
    )

    print(
        importance.head(20).to_string(
            index=False
        )
    )

    importance.to_csv(
        f"{RESULTS_DIR}/lightgbm_feature_importance.csv",
        index=False,
    )

    (
        shap_importance,
        shap_values,
        X_sample
    ) = run_shap_analysis(
        model,
        X_transformed
    )

    print(
        "\n=== Top SHAP Features ==="
    )

    print(
        shap_importance.head(20).to_string(
            index=False
        )
    )

    shap_importance.to_csv(
        f"{RESULTS_DIR}/shap_feature_importance.csv",
        index=False,
    )

    print(
        "\n=== Aggregated Feature Groups ==="
    )

    aggregated = aggregate_shipping_mode(
        shap_importance
    )

    print(
        aggregated.to_string(
            index=False
        )
    )

    aggregated.to_csv(
        f"{RESULTS_DIR}/shap_feature_groups.csv",
        index=False,
    )

    # Save SHAP values for reproducibility.
    shap_values_df = pd.DataFrame(
        shap_values,
        columns=X_sample.columns
    )

    shap_values_df.to_csv(
        f"{RESULTS_DIR}/shap_values_sample.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        "- results/lightgbm_feature_importance.csv"
    )
    print(
        "- results/shap_feature_importance.csv"
    )
    print(
        "- results/shap_feature_groups.csv"
    )
    print(
        "- results/shap_values_sample.csv"
    )


if __name__ == "__main__":
    main()