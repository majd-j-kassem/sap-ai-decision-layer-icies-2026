import pandas as pd

df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin1")

df["Supply_Deviation"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

print("\nSupply_Deviation:")
print(df["Supply_Deviation"].describe())
print(df["Supply_Deviation"].value_counts().sort_index())

print("\nDeviation vs Late_delivery_risk:")
print(pd.crosstab(df["Supply_Deviation"], df["Late_delivery_risk"]))

print("\nDeviation vs Delivery Status:")
print(pd.crosstab(df["Supply_Deviation"], df["Delivery Status"]))

print("Shape:", df.shape)
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(20))

for col in ["Delivery Status", "Late_delivery_risk"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

cols = ["Days for shipping (real)", "Days for shipment (scheduled)"]
print("\nStatistics:")
print(df[cols].describe())


print("\nScheduled shipping distribution:")
print(
    df["Days for shipment (scheduled)"]
    .value_counts()
    .sort_index()
)

print("\nReal shipping by scheduled time:")
print(
    pd.crosstab(
        df["Days for shipment (scheduled)"],
        df["Days for shipping (real)"]
    )
)

cols = [
    "Order Item Total",
    "Sales",
    "Benefit per order",
    "Order Profit Per Order"
]

print("\nCandidate continuous variables:")
print(df[cols].describe().T)
print("\nShipping mode vs scheduled days:")
print(pd.crosstab(
    df["Shipping Mode"],
    df["Days for shipment (scheduled)"]
))
print("\nReal shipping statistics by mode:")
print(
    df.groupby("Shipping Mode")["Days for shipping (real)"]
      .agg(["count", "mean", "std", "min", "max"])
      .round(3)
)
for col in ["Shipping Mode", "Market", "Customer Segment", "Order Region"]:
    print(f"\n{col} vs Late_delivery_risk:")
    print(pd.crosstab(df[col], df["Late_delivery_risk"], normalize="index").round(3))

candidate_cols = [
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Order Region",
    "Order Country",
    "Customer Country",
    "Department Name",
    "Category Name",
]

for col in candidate_cols:
    rates = df.groupby(col)["Late_delivery_risk"].mean().sort_values(ascending=False)
    print(f"\n{col} — late-risk range:")
    print(f"min={rates.min():.3f}, max={rates.max():.3f}")

# Check candidate predictors against the target
candidate_cols = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Order Item Quantity",
    "Order Item Total",
    "Sales",
    "Benefit per order",
    "Order Item Profit Ratio"
]

for col in candidate_cols:
    if col in df.columns:
        print(f"\n{col} vs Late_delivery_risk:")
        print(pd.crosstab(df[col], df["Late_delivery_risk"], normalize="index").round(3))

candidate_features = [
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Order Region",
    "Department Name",
    "Category Name",
    "Order Item Quantity",
    "Product Price"
]

print("\nCandidate features:")
print(df[candidate_features].describe(include="all").T)