# SAP-Oriented Explainable AI Decision Layer for IIoT-Driven Operations

## Overview

This repository contains the experimental prototype developed for ICIES 2026.

The work proposes an explainable human-in-the-loop AI decision layer that connects machine learning predictions with SAP-oriented workflow simulation.

The framework includes:

- Predictive ML models for operational risk estimation.
- Risk-based decision layer.
- Human review escalation mechanism.
- SAP workflow simulation.
- Interactive Streamlit dashboard.

---

## Project Structure


## Project Structure

```text
2026_ICIES/
├── data/                  # Dataset (not included)
├── models/                # Trained ML models
├── scripts_and_models/    # Experimental scripts
├── results/               # Experimental outputs
├── latex/                 # Paper source
└── dashboard/             # Interactive dashboard

---

## Dataset

## Download Dataset at:

The experiments use the DataCo Supply Chain Dataset.

The dataset is not included in this repository due to size and licensing considerations.

Download:

[DataCo Supply Chain Dataset](https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/b60060a2-e731-4745-8d51-3db158a1add7)

After downloading, place the file at:
data/DataCoSupplyChainDataset.csv

## Installation

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Run Decision API

From project root:

uvicorn api.decision_api:app --reload --app-dir src

API:

http://127.0.0.1:8000
Run Dashboard

Open another terminal:

streamlit run dashboard/sap_dashboard.py

Dashboard:

http://localhost:8501
Prototype Workflow
Operational Data
        |
        v
Machine Learning Model
        |
        v
Risk Assessment
        |
        v
Decision Layer
        |
        +---- Low Risk --> Automated Action
        |
        +---- Medium/High Risk --> Human Review
        |
        v
SAP-Oriented Workflow Simulation
Dataset

The experiments use the DataCo Supply Chain Dataset.

The dataset is excluded from the repository due to size and licensing considerations.

data/DataCoSupplyChainDataset.csv
Reproducibility

The repository provides:

Model training scripts.
Evaluation experiments.
Decision-layer experiments.
SAP workflow simulation.
Dashboard prototype.

| Stage | Script | Objective |
| :--- | :--- | :--- |
| **1. Data Understanding** | `01_data_profile.py` | Dataset profiling, quality inspection, and statistical overview |
| **2. Relationship Analysis** | `02_relationship_tests.py` | Analyze relationships between operational variables and target outcome |
| **3. Feature Engineering** | `03_feature_screening.py` | Identify relevant predictive features and remove weak variables |
| **4. Baseline Modeling** | `04_model_training.py` | Train baseline machine-learning models |
| **5. Model Diagnostics** | `05_model_diagnostics.py` | Evaluate model behavior, errors, and performance characteristics |
| **6. Feature Contribution Analysis** | `06_feature_ablation.py` | Measure the effect of removing features on model performance |
| **7. Hybrid Modeling** | `06_kalman_hybrid.py` | Develop Kalman-filter-enhanced hybrid prediction approach |
| **8. Statistical Validation** | `07_confounding_analysis.py`, `08_statistical_validation.py` | Validate robustness and statistical significance |
| **9. Advanced Models** | `09_xgboost_lightgbm.py` | Train and compare advanced gradient boosting models |
| **10. Explainability** | `10_model_interpretation.py` | Generate feature importance and explain model decisions |
| **11. Robustness Testing** | `11_robustness_analysis.py` | Evaluate stability across samples and seeds |
| **12. Decision Layer** | `12_decision_layer.py` | Transform predictions into risk-based operational decisions |
| **13. SAP Integration Prototype** | `13_sap_integration_prototype.py` | Simulate integration between AI outputs and SAP workflows |
| **14. Human-in-the-loop** | `14_human_in_the_loop.py` | Implement selective escalation for human review |
| **15. Decision Evaluation** | `17_decision_effectiveness.py` | Measure decision-layer effectiveness |
| **16. Calibration Analysis** | `18_calibration_analysis.py` | Evaluate prediction confidence calibration |
| **17. Economic Evaluation** | `19_cost_sensitive_analysis.py` | Analyze cost impact, optimization, and break-even points |
| **18. SAP Workflow Simulation** | `24_sap_workflow_simulation.py` | Demonstrate enterprise workflow execution logic |
| **19. Architecture Visualization** | `25_sap_architecture_diagram.py` | Generate SAP-oriented architecture representation |
| **20. End-to-End Validation** | `26_end_to_end_evaluation.py` | Validate complete framework performance |
| **21. Paper Preparation** | `29_results_visualization.py` | Generate figures, tables, and final manuscript results |