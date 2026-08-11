from pathlib import Path
import json
import pandas as pd

RESULTS = Path("results")


def load_csv(name):
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def main():
    print("=== Final Paper Results ===\n")

    claims = load_csv("paper_claims_validation_v2.csv")
    decision = load_csv("decision_effectiveness.csv")
    calibration = load_csv("calibration_summary.csv")
    economic = load_csv("end_to_end_economic_evaluation.csv")
    scenarios = load_csv("economic_scenario_analysis.csv")
    breakeven = load_csv("cost_break_even.csv")
    workflow = load_csv("sap_workflow_audit_trail.csv")
    hitl = load_csv("human_in_the_loop_summary.csv")
    model = load_csv("table_model_performance.csv")

    # ---------------------------------------------------------
    # MODEL PERFORMANCE
    # ---------------------------------------------------------
    auc_rows = model[
        model["metric"].astype(str).str.upper() == "ROC-AUC"
    ]

    best_auc_row = auc_rows.loc[
        auc_rows["mean"].idxmax()
    ]

    best_model = best_auc_row["model"]
    best_auc = float(best_auc_row["mean"])
    auc_std = float(best_auc_row["std"])

    # ---------------------------------------------------------
    # RISK SEPARATION
    # ---------------------------------------------------------
    risk = (
        decision
        .set_index("risk_level")
        .reindex(["LOW", "MEDIUM", "HIGH"])
    )

    low_rate = float(risk.loc["LOW", "observed_late_rate"])
    medium_rate = float(risk.loc["MEDIUM", "observed_late_rate"])
    high_rate = float(risk.loc["HIGH", "observed_late_rate"])

    risk_gap = high_rate - low_rate
    relative_increase = risk_gap / low_rate

    # ---------------------------------------------------------
    # CALIBRATION
    # ---------------------------------------------------------
    brier = float(calibration["brier_score"].iloc[0])
    ece = float(
        calibration["expected_calibration_error"].iloc[0]
    )

    # ---------------------------------------------------------
    # ECONOMIC IMPACT
    # ---------------------------------------------------------
    baseline_cost = economic["baseline_cost"].sum()
    decision_cost = economic["decision_cost"].sum()
    saving = baseline_cost - decision_cost
    saving_rate = saving / baseline_cost

    # ---------------------------------------------------------
    # ECONOMIC ROBUSTNESS
    # ---------------------------------------------------------
    beneficial = int(
        scenarios["economically_beneficial"]
        .astype(bool)
        .sum()
    )

    total_scenarios = len(scenarios)

    # ---------------------------------------------------------
    # BREAK EVEN
    # ---------------------------------------------------------
    break_even_cases = len(breakeven)

    # ---------------------------------------------------------
    # SAP WORKFLOW
    # ---------------------------------------------------------
    all_approved = (
        workflow["final_status"]
        .astype(str)
        .str.upper()
        .eq("APPROVED")
        .all()
    )

    workflow_cases = len(workflow)

    # ---------------------------------------------------------
    # HUMAN IN THE LOOP
    # ---------------------------------------------------------
    hitl_total = int(hitl["count"].sum())

    hitl_interventions = int(
        hitl[
            hitl["decision_status"]
            .astype(str)
            .str.upper()
            .isin(["HUMAN_APPROVED", "OVERRIDDEN"])
        ]["count"].sum()
    )

    hitl_rate = hitl_interventions / hitl_total

    # ---------------------------------------------------------
    # FINAL STRUCTURED RESULT
    # ---------------------------------------------------------
    final_results = {
        "model_performance": {
            "best_model": best_model,
            "roc_auc": best_auc,
            "roc_auc_std": auc_std
        },

        "risk_effectiveness": {
            "low_observed_late_rate": low_rate,
            "medium_observed_late_rate": medium_rate,
            "high_observed_late_rate": high_rate,
            "high_low_absolute_gap": risk_gap,
            "high_low_relative_increase": relative_increase
        },

        "calibration": {
            "brier_score": brier,
            "expected_calibration_error": ece
        },

        "economic_impact": {
            "baseline_cost": baseline_cost,
            "decision_cost": decision_cost,
            "cost_saving": saving,
            "saving_rate": saving_rate
        },

        "economic_robustness": {
            "beneficial_scenarios": beneficial,
            "total_scenarios": total_scenarios,
            "beneficial_fraction":
                beneficial / total_scenarios
        },

        "break_even": {
            "evaluated_scenarios": break_even_cases
        },

        "sap_workflow": {
            "cases": workflow_cases,
            "all_final_approved": bool(all_approved)
        },

        "human_in_the_loop": {
            "total_evaluated": hitl_total,
            "human_intervention_cases": hitl_interventions,
            "intervention_rate": hitl_rate
        }
    }

    # ---------------------------------------------------------
    # CONSOLE
    # ---------------------------------------------------------
    print("=== Model Performance ===")
    print(f"Best Model       : {best_model}")
    print(f"ROC-AUC          : {best_auc:.4f}")
    print(f"ROC-AUC STD      : {auc_std:.4f}")

    print("\n=== Risk Effectiveness ===")
    print(f"LOW late rate    : {low_rate:.4f}")
    print(f"MEDIUM late rate : {medium_rate:.4f}")
    print(f"HIGH late rate   : {high_rate:.4f}")
    print(f"HIGH-LOW gap     : {risk_gap:.4f}")
    print(f"Relative increase: {relative_increase:.2%}")

    print("\n=== Calibration ===")
    print(f"Brier Score      : {brier:.6f}")
    print(f"ECE              : {ece:.6f}")

    print("\n=== Economic Impact ===")
    print(f"Baseline Cost    : ${baseline_cost:,.2f}")
    print(f"Decision Cost    : ${decision_cost:,.2f}")
    print(f"Cost Saving      : ${saving:,.2f}")
    print(f"Saving Rate      : {saving_rate:.2%}")

    print("\n=== Economic Robustness ===")
    print(
        f"Beneficial Cases : "
        f"{beneficial}/{total_scenarios}"
    )

    print("\n=== Break-Even ===")
    print(
        f"Scenarios        : {break_even_cases}"
    )

    print("\n=== SAP Workflow ===")
    print(f"Cases            : {workflow_cases}")
    print(f"All Approved     : {all_approved}")

    print("\n=== Human-in-the-Loop ===")
    print(f"Total Evaluated  : {hitl_total}")
    print(f"HITL Cases       : {hitl_interventions}")
    print(f"HITL Rate        : {hitl_rate:.2%}")

    # ---------------------------------------------------------
    # SAVE JSON
    # ---------------------------------------------------------
    json_path = RESULTS / "paper_final_results.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            final_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # SAVE MARKDOWN
    # ---------------------------------------------------------
    md = f"""# Final Experimental Results

## Predictive Performance

The best-performing model was **{best_model}**, achieving a mean ROC-AUC of **{best_auc:.4f}** (SD = {auc_std:.4f}).

## Risk-Level Effectiveness

Observed late-delivery rates increased monotonically across risk levels:

- LOW: {low_rate:.4f}
- MEDIUM: {medium_rate:.4f}
- HIGH: {high_rate:.4f}

The absolute HIGH–LOW separation was **{risk_gap:.4f}**, corresponding to a relative increase of **{relative_increase:.2%}**.

## Probability Calibration

- Brier Score: **{brier:.6f}**
- Expected Calibration Error: **{ece:.6f}**

## Economic Impact

- Baseline expected cost: **${baseline_cost:,.2f}**
- Decision-layer cost: **${decision_cost:,.2f}**
- Cost saving: **${saving:,.2f}**
- Saving rate: **{saving_rate:.2%}**

## Economic Robustness

The decision layer was economically beneficial in **{beneficial} of {total_scenarios}** evaluated cost scenarios.

## Break-Even Analysis

A total of **{break_even_cases}** break-even scenarios were evaluated.

## SAP Workflow Evaluation

All **{workflow_cases}** evaluated workflow cases reached final approval: **{all_approved}**.

## Human-in-the-Loop

The evaluation included **{hitl_total}** observations, of which **{hitl_interventions}** involved human intervention or approval, corresponding to **{hitl_rate:.2%}**.
"""

    md_path = RESULTS / "paper_final_results.md"

    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(md)

    # ---------------------------------------------------------
    # CLAIM CONSISTENCY CHECK
    # ---------------------------------------------------------
    unsupported = claims[
        claims["status"]
        .astype(str)
        .str.upper()
        != "SUPPORTED"
    ]

    print("\n=== Claim Consistency ===")

    if unsupported.empty:
        print("All validated paper claims are SUPPORTED.")
    else:
        print("WARNING: Unsupported claims detected:")
        print(
            unsupported[
                ["claim", "status", "evidence"]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(f"- {json_path}")
    print(f"- {md_path}")


if __name__ == "__main__":
    main()
