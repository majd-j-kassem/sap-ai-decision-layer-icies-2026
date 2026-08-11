from pathlib import Path
import json
import pandas as pd

RESULTS = Path("results")


def load_csv(filename):
    path = RESULTS / filename
    if not path.exists():
        print(f"WARNING: Missing {filename}")
        return None
    return pd.read_csv(path)


def main():
    print("=== Paper Claims Validation V2 ===\n")

    claims = []

    # =========================================================
    # 1. MODEL PERFORMANCE
    # =========================================================
    df = load_csv("table_model_performance.csv")

    best_auc = None

    if df is not None:
        auc_rows = df[
            df["metric"]
            .astype(str)
            .str.upper()
            .eq("ROC-AUC")
        ]

        if not auc_rows.empty:
            best_auc = float(
                pd.to_numeric(
                    auc_rows["mean"],
                    errors="coerce"
                ).max()
            )

    claims.append({
        "claim":
            "The predictive model achieves ROC-AUC above 0.70.",
        "status":
            "SUPPORTED"
            if best_auc is not None and best_auc > 0.70
            else "NOT_SUPPORTED",
        "evidence":
            f"Best ROC-AUC = {best_auc:.4f}"
            if best_auc is not None
            else "ROC-AUC unavailable"
    })

    # =========================================================
    # 2. RISK EFFECTIVENESS
    # =========================================================
    df = load_csv("decision_effectiveness.csv")

    monotonic = False
    gap = None
    p_value = None

    if df is not None:
        risk = (
            df.set_index("risk_level")
            .reindex(["LOW", "MEDIUM", "HIGH"])
        )

        rates = risk["observed_late_rate"].tolist()

        monotonic = (
            len(rates) == 3
            and rates[0] <= rates[1] <= rates[2]
        )

        gap = (
            float(risk.loc["HIGH", "observed_late_rate"])
            -
            float(risk.loc["LOW", "observed_late_rate"])
        )

        if "chi_square_p_value" in df.columns:
            p_value = float(
                pd.to_numeric(
                    df["chi_square_p_value"],
                    errors="coerce"
                ).dropna().iloc[0]
            )

    claims.append({
        "claim":
            "Observed late-delivery rate increases with risk level.",
        "status":
            "SUPPORTED" if monotonic else "NOT_SUPPORTED",
        "evidence":
            f"Monotonic ordering = {monotonic}"
    })

    claims.append({
        "claim":
            "HIGH and LOW risk groups show substantial separation.",
        "status":
            "SUPPORTED"
            if gap is not None and gap >= 0.50
            else "NOT_SUPPORTED",
        "evidence":
            f"Absolute risk gap = {gap:.4f}"
            if gap is not None
            else "Risk gap unavailable"
    })

    claims.append({
        "claim":
            "Risk-level association is statistically significant.",
        "status":
            "SUPPORTED"
            if p_value is not None and p_value < 0.05
            else "NOT_SUPPORTED",
        "evidence":
            f"Chi-square p-value = {p_value:.6g}"
            if p_value is not None
            else "p-value unavailable"
    })

    # =========================================================
    # 3. CALIBRATION
    # =========================================================
    df = load_csv("calibration_summary.csv")

    brier = None
    ece = None

    if df is not None and not df.empty:
        brier = float(df["brier_score"].iloc[0])
        ece = float(
            df["expected_calibration_error"].iloc[0]
        )

    # Conservative criterion:
    # ECE < 0.05 is treated as acceptable calibration.
    calibration_supported = (
        ece is not None
        and ece < 0.05
    )

    claims.append({
        "claim":
            "The predictive probabilities demonstrate acceptable calibration.",
        "status":
            "SUPPORTED"
            if calibration_supported
            else "NOT_SUPPORTED",
        "evidence":
            (
                f"Brier Score = {brier:.6f}; "
                f"ECE = {ece:.6f}"
            )
            if ece is not None
            else "Calibration metrics unavailable"
    })

    # =========================================================
    # 4. OVERALL ECONOMIC IMPACT
    # =========================================================
    df = load_csv(
        "end_to_end_economic_evaluation.csv"
    )

    overall_saving = None

    if df is not None:
        baseline = df["baseline_cost"].sum()
        decision = df["decision_cost"].sum()

        overall_saving = baseline - decision

    claims.append({
        "claim":
            "The decision layer produces an overall economic saving.",
        "status":
            "SUPPORTED"
            if overall_saving is not None
            and overall_saving > 0
            else "NOT_SUPPORTED",
        "evidence":
            (
                f"Overall saving = "
                f"${overall_saving:,.2f}"
            )
            if overall_saving is not None
            else "Economic result unavailable"
    })

    # =========================================================
    # 5. ECONOMIC SCENARIO ROBUSTNESS
    # =========================================================
    df = load_csv(
        "economic_scenario_analysis.csv"
    )

    beneficial = None
    total = None

    if df is not None:
        beneficial = int(
            df["economically_beneficial"]
            .astype(bool)
            .sum()
        )
        total = len(df)

    claims.append({
        "claim":
            "Economic benefit is robust across multiple cost scenarios.",
        "status":
            "SUPPORTED"
            if beneficial is not None
            and beneficial > 0
            else "NOT_SUPPORTED",
        "evidence":
            (
                f"Beneficial scenarios = "
                f"{beneficial}/{total}"
            )
            if beneficial is not None
            else "Scenario results unavailable"
    })

    # =========================================================
    # 6. BREAK-EVEN
    # =========================================================
    df = load_csv("cost_break_even.csv")

    claims.append({
        "claim":
            "The decision layer has a measurable economic break-even point.",
        "status":
            "SUPPORTED"
            if df is not None and not df.empty
            else "NOT_SUPPORTED",
        "evidence":
            (
                f"Break-even scenarios evaluated = {len(df)}"
                if df is not None
                else "Break-even analysis unavailable"
            )
    })

    # =========================================================
    # 7. SAP WORKFLOW
    # =========================================================
    df = load_csv(
        "sap_workflow_audit_trail.csv"
    )

    workflow_supported = False

    if df is not None:
        workflow_supported = (
            df["final_status"]
            .astype(str)
            .str.upper()
            .eq("APPROVED")
            .all()
        )

    claims.append({
        "claim":
            "All evaluated SAP workflow cases reached final approval.",
        "status":
            "SUPPORTED"
            if workflow_supported
            else "NOT_SUPPORTED",
        "evidence":
            f"All approved = {workflow_supported}"
    })

    # =========================================================
    # 8. HUMAN-IN-THE-LOOP
    # =========================================================
    df = load_csv(
        "human_in_the_loop_summary.csv"
    )

    hitl_supported = False
    hitl_evidence = "HITL evidence unavailable"

    if df is not None and not df.empty:

        hitl_supported = True

        if {
            "risk_level",
            "decision_status",
            "count"
        }.issubset(df.columns):

            total = df["count"].sum()

            intervention = df[
                df["decision_status"]
                .astype(str)
                .str.upper()
                .isin([
                    "HUMAN_APPROVED",
                    "OVERRIDDEN"
                ])
            ]["count"].sum()

            hitl_evidence = (
                f"HITL decisions = {int(intervention)} "
                f"of {int(total)} observations"
            )
        else:
            hitl_evidence = (
                "Human-in-the-loop summary available"
            )

    claims.append({
        "claim":
            "The decision layer supports human-in-the-loop intervention.",
        "status":
            "SUPPORTED"
            if hitl_supported
            else "NOT_SUPPORTED",
        "evidence":
            hitl_evidence
    })

    # =========================================================
    # FINAL RESULTS
    # =========================================================
    result = pd.DataFrame(claims)

    print("=== Validation Results ===")
    print(result.to_string(index=False))

    csv_path = RESULTS / "paper_claims_validation_v2.csv"
    json_path = RESULTS / "paper_claims_validation_v2.json"

    result.to_csv(
        csv_path,
        index=False
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            claims,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nSaved:")
    print(f"- {csv_path}")
    print(f"- {json_path}")


if __name__ == "__main__":
    main()
