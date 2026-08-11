import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RESULTS = "results"
DATA_PATH = "data/DataCoSupplyChainDataset.csv"
MODEL_PATH = "models/lightgbm_model.joblib"

os.makedirs(RESULTS, exist_ok=True)


def save_plot(fig, name):
    fig.savefig(
        f"{RESULTS}/{name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        f"{RESULTS}/{name}.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


def main():

    # 1. Risk effectiveness
    df = pd.read_csv(
        f"{RESULTS}/decision_effectiveness.csv"
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        df["risk_level"],
        df["observed_late_rate"] * 100
    )

    ax.set_title(
        "Observed Late-Delivery Rate by Risk Level"
    )
    ax.set_ylabel("Late-delivery rate (%)")
    ax.set_xlabel("Risk level")

    save_plot(fig, "risk_effectiveness")


    # 2. Calibration Reliability Diagram
    cal = pd.read_csv(
        f"{RESULTS}/calibration_analysis.csv"
    )

    pred_col = next(
        c for c in cal.columns
        if "pred" in c.lower()
    )

    obs_col = next(
        c for c in cal.columns
        if "observ" in c.lower()
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        cal[pred_col],
        cal[obs_col],
        marker="o",
        linewidth=2,
        label="LightGBM calibration"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration (y=x)"
    )

    ax.set_title(
        "Calibration Reliability Diagram"
    )
    ax.set_xlabel(
        "Mean predicted probability"
    )
    ax.set_ylabel(
        "Observed frequency"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3)
    ax.legend()

    save_plot(
        fig,
        "calibration_curve"
    )


    # 3. Economic impact
    eco = pd.read_csv(
        f"{RESULTS}/end_to_end_economic_evaluation.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(eco))

    ax.bar(
        [i - 0.2 for i in x],
        eco["baseline_cost"],
        width=0.4,
        label="Baseline cost"
    )

    ax.bar(
        [i + 0.2 for i in x],
        eco["decision_cost"],
        width=0.4,
        label="Decision cost"
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(eco["risk_level"])
    ax.set_ylabel("Cost")
    ax.set_title(
        "Baseline vs Decision Cost by Risk Level"
    )
    ax.legend()

    save_plot(
        fig,
        "economic_impact"
    )


    # 4. Cost sensitivity
    sens = pd.read_csv(
        f"{RESULTS}/economic_scenario_analysis.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for cost in sorted(
        sens["late_delivery_cost"].unique()
    ):
        subset = sens[
            sens["late_delivery_cost"] == cost
        ]

        ax.plot(
            subset["intervention_cost"],
            subset["saving_rate"] * 100,
            marker="o",
            label=f"Late cost = ${cost}"
        )

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.set_title(
        "Economic Sensitivity to Intervention Cost"
    )
    ax.set_xlabel(
        "Intervention cost per order ($)"
    )
    ax.set_ylabel(
        "Saving rate (%)"
    )
    ax.legend()

    save_plot(
        fig,
        "cost_sensitivity"
    )


    # 5. Confusion Matrix - LightGBM
    print("Generating LightGBM confusion matrix...")

    data = pd.read_csv(
        DATA_PATH,
        encoding="latin1"
    )

    target = "Late_delivery_risk"

    features = [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Order Item Quantity",
        "Product Price",
    ]

    X = data[features]
    y = data[target]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(
        MODEL_PATH
    )

    y_pred = model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "On-time",
            "Late"
        ]
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(
        "LightGBM Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "Actual class"
    )

    save_plot(
        fig,
        "confusion_matrix"
    )

        # 6. Feature Importance - LightGBM
    print("Generating LightGBM feature importance...")

    preprocessor = model.named_steps["preprocessor"]
    lightgbm = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = lightgbm.feature_importances_

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    importance = importance.sort_values(
        "importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        importance["feature"],
        importance["importance"]
    )

    ax.set_title("LightGBM Feature Importance")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")

    save_plot(fig, "feature_importance")

    print("- feature_importance.png/pdf")
    # 7. Human-in-the-Loop Decision Flow
    print("Generating human-in-the-loop decision flow...")

    hitl = pd.read_csv(
        f"{RESULTS}/human_in_the_loop_results.csv"
    )

    flow_counts = (
        hitl.groupby(["risk_level", "decision_status"])
        .size()
        .reset_index(name="count")
    )

    pivot = flow_counts.pivot(
        index="risk_level",
        columns="decision_status",
        values="count"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title(
        "Human-in-the-Loop Decision Outcomes"
    )
    ax.set_xlabel("Risk level")
    ax.set_ylabel("Number of orders")
    ax.legend(title="Decision status")

    plt.xticks(rotation=0)
    plt.tight_layout()

    save_plot(
        fig,
        "human_in_the_loop_flow"
    )


    # 8. Before vs After Framework Intervention
    print("Generating before-vs-after intervention comparison...")

    before_after = pd.read_csv(
        f"{RESULTS}/end_to_end_economic_evaluation.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(before_after))

    ax.bar(
        [i - 0.2 for i in x],
        before_after["baseline_cost"],
        width=0.4,
        label="Without framework"
    )

    ax.bar(
        [i + 0.2 for i in x],
        before_after["decision_cost"],
        width=0.4,
        label="With framework"
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        before_after["risk_level"]
    )

    ax.set_ylabel("Expected cost")
    ax.set_xlabel("Risk level")

    ax.set_title(
        "Before vs After Framework Intervention"
    )

    ax.legend()

    save_plot(
        fig,
        "before_after_intervention"
    )

    print("- human_in_the_loop_flow.png/pdf")
    print("- before_after_intervention.png/pdf")


    print("Saved:")
    print("- risk_effectiveness.png/pdf")
    print("- calibration_curve.png/pdf")
    print("- economic_impact.png/pdf")
    print("- cost_sensitivity.png/pdf")
    print("- confusion_matrix.png/pdf")
    print("- human_in_the_loop_flow.png/pdf")
    print("- before_after_intervention.png/pdf")

    


if __name__ == "__main__":
    main()
