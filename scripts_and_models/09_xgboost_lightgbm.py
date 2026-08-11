import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


DATA_PATH = "data/DataCoSupplyChainDataset.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"

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


def build_xgboost():
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
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


def evaluate_holdout(
    name,
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
):
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    result = {
        "model": name,
        "accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
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
        "pr_auc": average_precision_score(
            y_test,
            y_prob,
        ),
    }

    print(f"\n=== {name} Holdout Results ===")
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
    print(
        f"PR-AUC   : {result['pr_auc']:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    return result


def repeated_cv(
    name,
    X,
    y,
    model_builder,
    n_splits=15,
):
    print(f"\n=== {name} Repeated CV ===")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    scores = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    model_builder(),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_prob,
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob,
        )

        scores.append(
            {
                "model": name,
                "fold": fold,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

        print(
            f"Fold {fold:02d} | "
            f"ROC-AUC={roc_auc:.6f} | "
            f"PR-AUC={pr_auc:.6f}"
        )

    return pd.DataFrame(scores)


def main():
    print("=== XGBoost / LightGBM Evaluation ===")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    X, y = load_data()

    print(f"Dataset shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "XGBoost": build_xgboost,
        "LightGBM": build_lightgbm,
    }

    holdout_results = []
    cv_results = []

    trained_pipelines = {}

    for name, builder in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    builder(),
                ),
            ]
        )

        result = evaluate_holdout(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        holdout_results.append(result)

        cv_df = repeated_cv(
            name,
            X,
            y,
            builder,
            n_splits=15,
        )

        cv_results.append(cv_df)

        # Retrain on the complete dataset
        # for deployment/integration.
        final_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    builder(),
                ),
            ]
        )

        final_pipeline.fit(X, y)

        trained_pipelines[name] = final_pipeline

        model_filename = (
            f"{MODELS_DIR}/"
            f"{name.lower().replace(' ', '_')}"
            f"_model.joblib"
        )

        joblib.dump(
            final_pipeline,
            model_filename,
        )

        print(
            f"Saved model: {model_filename}"
        )

    holdout_df = pd.DataFrame(
        holdout_results
    )

    cv_df = pd.concat(
        cv_results,
        ignore_index=True,
    )

    cv_summary = (
        cv_df
        .groupby("model")
        .agg(
            cv_roc_auc_mean=("roc_auc", "mean"),
            cv_roc_auc_std=("roc_auc", "std"),
            cv_pr_auc_mean=("pr_auc", "mean"),
            cv_pr_auc_std=("pr_auc", "std"),
        )
        .reset_index()
    )

    comparison = holdout_df.merge(
        cv_summary,
        on="model",
        how="left",
    )

    print("\n=== Holdout Comparison ===")
    print(
        comparison.to_string(
            index=False
        )
    )

    print("\n=== Cross-Validation Comparison ===")
    print(
        cv_summary.to_string(
            index=False
        )
    )

    comparison.to_csv(
        f"{RESULTS_DIR}/"
        "xgboost_lightgbm_results.csv",
        index=False,
    )

    cv_df.to_csv(
        f"{RESULTS_DIR}/"
        "xgboost_lightgbm_cv.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        "- results/xgboost_lightgbm_results.csv"
    )
    print(
        "- results/xgboost_lightgbm_cv.csv"
    )
    print(
        "- models/xgboost_model.joblib"
    )
    print(
        "- models/lightgbm_model.joblib"
    )


if __name__ == "__main__":
    main()

