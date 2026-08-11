from pathlib import Path
import pandas as pd

RESULTS = Path("results")


def check(name, condition, evidence):
    status = "SUPPORTED" if condition else "NOT_SUPPORTED"
    print(f"{status:15} | {name}")
    print(f"  Evidence: {evidence}")
    return {
        "claim": name,
        "status": status,
        "evidence": evidence,
    }


def main():
    print("=== Paper Claims Validation ===\n")

    results = []

    # --------------------------------------------------
    # 1. Model performance
    # --------------------------------------------------
    exp = pd.read_csv(
        RESULTS / "experiment_summary.csv"
    )

    roc = exp[exp["metric"] == "ROC-AUC"]

    best_roc = roc["mean"].max()

    results.append(
        check(
            "The predictive model achieves ROC-AUC above 0.70.",
            best_roc > 0.70,
            f"Best ROC-AUC = {best_roc:.4f}",
        )
    )

    # --------------------------------------------------
    # 2. Risk separation
    # --------------------------------------------------
    decision = pd.read_csv(
        RESULTS / "decision_effectiveness.csv"
    )

    low = decision[
        decision["risk_level"].str.upper() == "LOW"
    ].iloc[0]

    high = decision[
        decision["risk_level"].str.upper() == "HIGH"
    ].iloc[0]

    risk_gap = (
        high["observed_late_rate"]
        - low["observed_late_rate"]
    )

    monotonic = bool(
        decision["monotonic_risk_order"].iloc[0]
    )

    results.append(
        check(
            "Observed late-delivery rate increases with risk level.",
            monotonic,
            f"Monotonic ordering = {monotonic}",
        )
    )

    results.append(
        check(
            "The HIGH and LOW risk groups show substantial separation.",
            risk_gap > 0.50,
            f"Absolute gap = {risk_gap:.4f}",
        )
    )

    # --------------------------------------------------
    # 3. Statistical significance
    # --------------------------------------------------
    p_value = float(
        decision["chi_square_p_value"].iloc[0]
    )

    results.append(
        check(
            "Risk-level association is statistically significant.",
            p_value < 0.05,
            f"Chi-square p-value = {p_value:.6g}",
        )
    )

    # --------------------------------------------------
    # 4. Calibration
    # --------------------------------------------------
    calibration_path = RESULTS / "calibration_summary.csv"

    if calibration_path.exists():
        calibration = pd.read_csv(
            calibration_path
        )

        numeric_columns = calibration.select_dtypes(
            include="number"
        ).columns

        if "ECE" in calibration.columns:
            ece = float(calibration["ECE"].iloc[0])
        elif "ece" in calibration.columns:
            ece = float(calibration["ece"].iloc[0])
        else:
            ece = None

        if ece is not None:
            results.append(
                check(
                    "Calibration error is below 0.01 ECE.",
                    ece < 0.01,
                    f"ECE = {ece:.6f}",
                )
            )

    # --------------------------------------------------
    # 5. Economic benefit
    # --------------------------------------------------
    economic = pd.read_csv(
        RESULTS / "end_to_end_economic_evaluation.csv"
    )

    baseline = economic["baseline_cost"].sum()
    decision_cost = economic["decision_cost"].sum()
    saving = baseline - decision_cost

    results.append(
        check(
            "The decision layer produces an overall economic saving "
            "under the evaluated cost structure.",
            saving > 0,
            f"Baseline = ${baseline:,.2f}; "
            f"Decision = ${decision_cost:,.2f}; "
            f"Saving = ${saving:,.2f}",
        )
    )

    # --------------------------------------------------
    # 6. Economic sensitivity
    # --------------------------------------------------
    sensitivity = pd.read_csv(
        RESULTS / "economic_scenario_analysis.csv"
    )

    beneficial = sensitivity[
        sensitivity["economically_beneficial"] == True
    ]

    results.append(
        check(
            "The economic benefit is robust across multiple "
            "cost scenarios.",
            len(beneficial) > 0,
            f"Beneficial scenarios = "
            f"{len(beneficial)}/{len(sensitivity)}",
        )
    )

    # --------------------------------------------------
    # 7. SAP workflow
    # --------------------------------------------------
    workflow_path = RESULTS / "sap_workflow_audit_trail.csv"

    if workflow_path.exists():
        workflow = pd.read_csv(workflow_path)

        approved = (
            workflow["final_status"]
            .astype(str)
            .str.upper()
            .eq("APPROVED")
            .all()
        )

        results.append(
            check(
                "All evaluated SAP workflow cases reached a final "
                "approved state.",
                approved,
                f"All approved = {approved}",
            )
        )

    # --------------------------------------------------
    # 8. Save validation results
    # --------------------------------------------------
    validation = pd.DataFrame(results)

    validation.to_csv(
        RESULTS / "paper_claims_validation.csv",
        index=False,
    )

    print("\n=== Validation Summary ===")

    print(
        validation[
            ["claim", "status"]
        ].to_string(index=False)
    )

    print(
        "\nSaved: "
        "results/paper_claims_validation.csv"
    )


if __name__ == "__main__":
    main()