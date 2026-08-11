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