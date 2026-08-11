import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = "data/DataCoSupplyChainDataset.csv"
RESULTS_PATH = "results/feature_ablation.csv"
TARGET = "Late_delivery_risk"


FEATURE_GROUPS = {
    "All Features": {
        "categorical": ["Shipping Mode", "Market", "Order Region"],
        "numeric": ["Order Item Quantity", "Product Price"],
    },
    "Shipping Mode Only": {
        "categorical": ["Shipping Mode"],
        "numeric": [],
    },
    "Without Shipping Mode": {
        "categorical": ["Market", "Order Region"],
        "numeric": ["Order Item Quantity", "Product Price"],
    },
    "Numeric Only": {
        "categorical": [],
        "numeric": ["Order Item Quantity", "Product Price"],
    },
}


MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "CART": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


def build_preprocessor(categorical_features, numeric_features):
    transformers = []

    if categorical_features:
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

        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    return ColumnTransformer(transformers=transformers)


def main():
    print("=== Feature Ablation Study ===")

    df = pd.read_csv(
        DATA_PATH,
        encoding="ISO-8859-1",
    )

    results = []

    for group_name, feature_config in FEATURE_GROUPS.items():

        categorical_features = feature_config["categorical"]
        numeric_features = feature_config["numeric"]

        features = categorical_features + numeric_features

        X = df[features].copy()
        y = df[TARGET].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        print(f"\n=== Feature Group: {group_name} ===")
        print(f"Features: {features}")

        for model_name, estimator in MODELS.items():

            preprocessor = build_preprocessor(
                categorical_features,
                numeric_features,
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            result = {
                "feature_group": group_name,
                "model": model_name,
                "n_features": len(features),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "roc_auc": roc_auc_score(
                    y_test,
                    y_prob,
                ),
            }

            results.append(result)

            print(
                f"{model_name:20s} "
                f"ROC-AUC={result['roc_auc']:.4f} "
                f"F1={result['f1']:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="roc_auc",
        ascending=False,
    )

    os.makedirs("results", exist_ok=True)

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\n=== Ablation Results ===")
    print(results_df.to_string(index=False))

    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()