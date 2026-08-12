import os
import pandas as pd
from scipy.stats import chi2_contingency
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "DataCoSupplyChainDataset.csv"
RESULTS_PATH = BASE_DIR / "results"

os.makedirs(RESULTS_PATH, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH, encoding="latin1")

# Create supply deviation diagnostic variable
df["Supply_Deviation"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

# =====================================================
# Dataset profile
# =====================================================

profile = pd.DataFrame({
    "Metric": [
        "Number of observations",
        "Number of variables"
    ],
    "Value": [
        df.shape[0],
        df.shape[1]
    ]
})

profile.to_csv(
    f"{RESULTS_PATH}/dataset_profile.csv",
    index=False
)


# =====================================================
# Target distribution
# =====================================================

target_distribution = (
    df["Late_delivery_risk"]
    .value_counts()
    .rename_axis("Class")
    .reset_index(name="Count")
)

target_distribution["Percentage"] = (
    target_distribution["Count"] / len(df) * 100
).round(2)

target_distribution.to_csv(
    f"{RESULTS_PATH}/target_distribution.csv",
    index=False
)


# =====================================================
# Missing values
# =====================================================

missing = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

missing.columns = [
    "Feature",
    "Missing_Count"
]

missing.to_csv(
    f"{RESULTS_PATH}/missing_values.csv",
    index=False
)


# =====================================================
# Descriptive statistics
# =====================================================

numeric_cols = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Order Item Total",
    "Sales",
    "Benefit per order",
    "Order Profit Per Order",
    "Order Item Quantity",
    "Product Price"
]

numeric_cols = [
    c for c in numeric_cols
    if c in df.columns
]

df[numeric_cols].describe().T.to_csv(
    f"{RESULTS_PATH}/descriptive_statistics.csv"
)


# =====================================================
# Categorical distribution
# =====================================================

categorical_cols = [
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Order Region",
    "Department Name",
    "Category Name"
]

for col in categorical_cols:

    if col in df.columns:

        distribution = (
            df[col]
            .value_counts()
            .reset_index()
        )

        distribution.columns = [
            col,
            "Count"
        ]

        distribution.to_csv(
            f"{RESULTS_PATH}/{col.replace(' ','_')}_distribution.csv",
            index=False
        )


# =====================================================
# Chi-square relationship tests
# =====================================================

chi_results = []

for col in categorical_cols:

    if col in df.columns:

        table = pd.crosstab(
            df[col],
            df["Late_delivery_risk"]
        )

        chi2, p, dof, expected = chi2_contingency(table)

        chi_results.append({
            "Feature": col,
            "Chi_square": chi2,
            "p_value": p,
            "Degrees_of_freedom": dof
        })


chi_results_df = pd.DataFrame(chi_results)

chi_results_df.to_csv(
    f"{RESULTS_PATH}/chi_square_results.csv",
    index=False
)


# =====================================================
# Late risk rates by operational attributes
# =====================================================

for col in categorical_cols:

    if col in df.columns:

        rates = (
            df.groupby(col)["Late_delivery_risk"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        rates.columns = [
            col,
            "Late_Risk_Rate"
        ]

        rates.to_csv(
            f"{RESULTS_PATH}/{col.replace(' ','_')}_late_risk_rates.csv",
            index=False
        )


# =====================================================
# Shipping performance baseline
# =====================================================

shipping_stats = (
    df.groupby("Shipping Mode")
    ["Days for shipping (real)"]
    .agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "max"
        ]
    )
    .round(3)
)

shipping_stats.to_csv(
    f"{RESULTS_PATH}/shipping_mode_performance.csv"
)


# =====================================================
# Supply deviation analysis
# Diagnostic only - not predictive feature
# =====================================================

df["Supply_Deviation"].describe().to_csv(
    f"{RESULTS_PATH}/supply_deviation_statistics.csv"
)

pd.crosstab(
    df["Supply_Deviation"],
    df["Late_delivery_risk"]
).to_csv(
    f"{RESULTS_PATH}/supply_deviation_vs_late_risk.csv"
)
target_distribution.to_latex(
    RESULTS_PATH / "target_distribution_table.tex",
    index=False,
    escape=True
)
chi_results_df.to_latex(
    RESULTS_PATH / "chi_square_results_table.tex",
    index=False,
    escape=True
)
print("Baseline statistical assessment completed.")
print(f"Results saved to: {RESULTS_PATH}")
