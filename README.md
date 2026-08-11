# An Explainable Human-in-the-loop AI Decision Layer for SAP ERP Integration Using IIoT Data

## Overview
This repository contains the experimental implementation of an AI-based decision layer designed to bridge machine learning predictions with SAP-oriented enterprise workflows.

The framework integrates:
- Predictive analytics
- Risk-based decision logic
- Human-in-the-loop control
- SAP workflow simulation
- Economic impact evaluation

## Project Structure


2026_ICIES/
├── data/ # Dataset (not included)
├── models/ # Trained ML models
├── scripts_and_models/ # Experimental scripts
├── results/ # Experimental outputs
├── latex/ # Paper source
└── dashboard/ # Interactive dashboard


## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Running the Decision API
cd scripts_and_models
uvicorn decision_api:app --reload

API endpoint:

http://127.0.0.1:8000
Running the Dashboard
cd scripts_and_models
streamlit run sap_dashboard.py

Dashboard:

http://localhost:8501
Models

The repository includes trained:

XGBoost model
LightGBM model
Research Outputs

The results/ directory contains:

Model evaluation
Calibration analysis
Robustness analysis
Human-in-the-loop evaluation
SAP workflow simulation
Economic impact analysis
Paper

The IEEE conference paper source is available in:

latex/main.tex