import pandas as pd
from scipy.stats import chi2_contingency

DATA_PATH = "data/DataCoSupplyChainDataset.csv"

# Load dataset using the actual file encoding
df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

# Reconstruct derived variable used in the profiling stage
df["Supply_Deviation"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

# ---------------------------------------------------------
# 1. Categorical relationship: Supply Deviation vs Late Risk
# ---------------------------------------------------------

contingency = pd.crosstab(
    df["Supply_Deviation"],
    df["Late_delivery_risk"]
)

chi2, p_value, dof, expected = chi2_contingency(contingency)

print("\n=== Supply Deviation vs Late Delivery Risk ===")
print(contingency)
print(f"Chi-square = {chi2:.4f}")
print(f"Degrees of freedom = {dof}")
print(f"p-value = {p_value:.6e}")

# ---------------------------------------------------------
# 2. Supply Deviation vs Delivery Status
# ---------------------------------------------------------

contingency_status = pd.crosstab(
    df["Supply_Deviation"],
    df["Delivery Status"]
)

chi2_status, p_status, dof_status, expected_status = chi2_contingency(
    contingency_status
)

print("\n=== Supply Deviation vs Delivery Status ===")
print(contingency_status)
print(f"Chi-square = {chi2_status:.4f}")
print(f"Degrees of freedom = {dof_status}")
print(f"p-value = {p_status:.6e}")

# ---------------------------------------------------------
# 3. Shipping Mode vs Late Delivery Risk
# ---------------------------------------------------------

contingency_mode = pd.crosstab(
    df["Shipping Mode"],
    df["Late_delivery_risk"]
)

chi2_mode, p_mode, dof_mode, expected_mode = chi2_contingency(
    contingency_mode
)

print("\n=== Shipping Mode vs Late Delivery Risk ===")
print(contingency_mode)
print(f"Chi-square = {chi2_mode:.4f}")
print(f"Degrees of freedom = {dof_mode}")
print(f"p-value = {p_mode:.6e}")

# ---------------------------------------------------------
# 4. Scheduled Shipping Days vs Late Delivery Risk
# ---------------------------------------------------------

contingency_schedule = pd.crosstab(
    df["Days for shipment (scheduled)"],
    df["Late_delivery_risk"]
)

chi2_schedule, p_schedule, dof_schedule, expected_schedule = chi2_contingency(
    contingency_schedule
)

print("\n=== Scheduled Shipping Days vs Late Delivery Risk ===")
print(contingency_schedule)
print(f"Chi-square = {chi2_schedule:.4f}")
print(f"Degrees of freedom = {dof_schedule}")
print(f"p-value = {p_schedule:.6e}")

# ---------------------------------------------------------
# 5. Categorical candidate features
# ---------------------------------------------------------

categorical_features = [
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Order Region",
    "Department Name",
    "Category Name",
]

print("\n=== Candidate Feature Association Tests ===")

for feature in categorical_features:
    contingency = pd.crosstab(
        df[feature],
        df["Late_delivery_risk"]
    )

    chi2, p_value, dof, expected = chi2_contingency(contingency)

    print(
        f"{feature:25s} "
        f"chi2={chi2:12.4f} "
        f"dof={dof:3d} "
        f"p={p_value:.6e}"
    )