import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = "data/DataCoSupplyChainDataset.csv"
TARGET = "Late_delivery_risk"

CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Order Region",
    "Department Name",
    "Category Name",
]

NUMERIC_FEATURES = [
    "Order Item Quantity",
    "Product Price",
]

# Excluded because they are post-outcome or directly derived from delivery outcome.
LEAKAGE_FEATURES = [
    "Supply_Deviation",
    "Days for shipping (real)",
    "Delivery Status",
    "Late_delivery_risk",
]

df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

required = [TARGET] + CATEGORICAL_FEATURES + NUMERIC_FEATURES
missing = [c for c in required if c not in df.columns]

if missing:
    raise KeyError(f"Missing required columns: {missing}")

df = df[required].copy()
df = df.dropna(subset=[TARGET])

y = df[TARGET].astype(int)

# Stratified development sample for feature screening
train_idx, _ = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=42,
    stratify=y,
)

train = df.iloc[train_idx].copy()
y_train = train[TARGET].astype(int)

results = []

# ============================================================
# CATEGORICAL FEATURES
# ============================================================

for feature in CATEGORICAL_FEATURES:

    x = train[feature].fillna("__MISSING__").astype(str)

    table = pd.crosstab(x, y_train)

    chi2, p_value, dof, _ = chi2_contingency(table)

    n = table.to_numpy().sum()
    rows, columns = table.shape

    denominator = min(rows - 1, columns - 1)

    cramers_v = (
        np.sqrt((chi2 / n) / denominator)
        if denominator > 0
        else 0.0
    )

    codes, _ = pd.factorize(x)

    mi = mutual_info_classif(
        codes.reshape(-1, 1),
        y_train.to_numpy(),
        discrete_features=True,
        random_state=42,
    )[0]

    results.append({
        "feature": feature,
        "type": "categorical",
        "chi2": chi2,
        "p_value": p_value,
        "cramers_v": cramers_v,
        "mutual_information": mi,
    })


# ============================================================
# NUMERIC FEATURES
# ============================================================

for feature in NUMERIC_FEATURES:

    x = pd.to_numeric(
        train[feature],
        errors="coerce"
    )

    x = x.fillna(x.median())

    correlation, p_value = pointbiserialr(
        y_train,
        x
    )

    mi = mutual_info_classif(
        x.to_numpy().reshape(-1, 1),
        y_train.to_numpy(),
        discrete_features=False,
        random_state=42,
    )[0]

    results.append({
        "feature": feature,
        "type": "numeric",
        "chi2": np.nan,
        "p_value": p_value,
        "cramers_v": abs(correlation),
        "mutual_information": mi,
    })


# ============================================================
# RANK FEATURES
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["mutual_information", "cramers_v"],
    ascending=False,
).reset_index(drop=True)

results_df["rank"] = np.arange(
    1,
    len(results_df) + 1
)

results_df["significant_at_0_05"] = (
    results_df["p_value"] < 0.05
)


# ============================================================
# SAVE RESULTS
# ============================================================

Path("results").mkdir(
    parents=True,
    exist_ok=True
)

output_path = "results/feature_screening.csv"

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n=== Predictive Feature Screening ===")

print(
    results_df.to_string(index=False)
)

print("\n=== Leakage Features Excluded ===")

for feature in LEAKAGE_FEATURES:
    print(f"- {feature}")

print(f"\nSaved: {output_path}")