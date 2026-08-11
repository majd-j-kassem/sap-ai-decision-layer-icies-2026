import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/DataCoSupplyChainDataset.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load DataCo dataset using its actual encoding
df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

# Target
target = "Late_delivery_risk"

# Features selected during screening
categorical_features = [
    "Shipping Mode",
    "Order Region",
    "Market",
]

numeric_features = [
    "Order Item Quantity",
    "Product Price",
]

features = categorical_features + numeric_features

X = df[features].copy()
y = df[target].astype(int)

# Preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_features),
    ("num", numeric_pipeline, numeric_features),
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "CART": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

diagnostic_results = []

print("\n=== Model Diagnostics ===")
print(f"Dataset shape: {X.shape}")
print(f"Features: {features}")

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    print(f"\n=== {name} ===")

    # Cross-validation ROC-AUC
    cv_auc = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    # Cross-validation PR-AUC
    cv_pr_auc = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
    )

    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    print(
        f"CV ROC-AUC: {cv_auc.mean():.4f} "
        f"+/- {cv_auc.std():.4f}"
    )

    print(
        f"CV PR-AUC : {cv_pr_auc.mean():.4f} "
        f"+/- {cv_pr_auc.std():.4f}"
    )

    # Threshold analysis
    print("\nThreshold analysis:")

    thresholds = np.arange(0.20, 0.81, 0.05)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0,
        )

        diagnostic_results.append({
            "model": name,
            "threshold": round(threshold, 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "cv_roc_auc_mean": cv_auc.mean(),
            "cv_roc_auc_std": cv_auc.std(),
            "cv_pr_auc_mean": cv_pr_auc.mean(),
            "cv_pr_auc_std": cv_pr_auc.std(),
        })

        print(
            f"threshold={threshold:.2f} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"F1={f1:.4f}"
        )

    # Best F1 threshold
    model_results = [
        r for r in diagnostic_results
        if r["model"] == name
    ]

    best = max(
        model_results,
        key=lambda x: x["f1"],
    )

    best_predictions = (
        probabilities >= best["threshold"]
    ).astype(int)

    cm = confusion_matrix(
        y,
        best_predictions,
    )

    print("\nBest threshold:")
    print(best["threshold"])

    print("\nBest F1:")
    print(f"{best['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # Feature importance
    if name in ["CART", "Random Forest"]:

        model_step = pipeline.named_steps["model"]
        preprocessor_step = pipeline.named_steps["preprocessor"]

        feature_names = (
            preprocessor_step
            .get_feature_names_out()
        )

        importances = model_step.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values(
            "importance",
            ascending=False,
        )

        importance_path = os.path.join(
            RESULTS_DIR,
            f"{name.lower().replace(' ', '_')}_feature_importance.csv",
        )

        importance_df.to_csv(
            importance_path,
            index=False,
        )

        print("\nTop feature importances:")

        print(
            importance_df.head(15).to_string(
                index=False
            )
        )

# Save complete diagnostics
diagnostics_df = pd.DataFrame(
    diagnostic_results
)

diagnostics_path = os.path.join(
    RESULTS_DIR,
    "model_diagnostics.csv",
)

diagnostics_df.to_csv(
    diagnostics_path,
    index=False,
)

# Best threshold summary
best_thresholds = (
    diagnostics_df
    .sort_values(
        ["model", "f1"],
        ascending=[True, False],
    )
    .groupby("model")
    .head(1)
)

best_threshold_path = os.path.join(
    RESULTS_DIR,
    "best_thresholds.csv",
)

best_thresholds.to_csv(
    best_threshold_path,
    index=False,
)

print("\n=== Diagnostic Summary ===")
print(
    best_thresholds[
        [
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(f"- {diagnostics_path}")
print(f"- {best_threshold_path}")
print("- Feature importance files for CART and Random Forest")