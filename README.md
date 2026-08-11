# Explainable Human-in-the-loop AI Decision Layer for SAP-Oriented Workflow Integration

## Overview

This repository contains the experimental implementation of an AI-driven decision layer that integrates:

- Logistics risk prediction
- Machine learning models
- Probability calibration
- Explainable risk stratification
- Human-in-the-loop decision control
- SAP-oriented workflow simulation
- Cost-sensitive economic evaluation

## Dataset

The experiments use the DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS dataset.

Due to size and licensing considerations, the raw dataset is not included.

## Project Structure


2026_ICIES/
├── latex/ # IEEE paper source
├── scripts_and_models/ # Experiment scripts
├── models/ # Trained ML models
├── results/ # Generated experiment results
├── dashboard/ # Decision dashboard
└── data/ # Dataset location


## Main Components

### Predictive Layer
- Logistic Regression
- CART
- Random Forest
- XGBoost
- LightGBM

### Decision Layer
Transforms prediction probabilities into:

- LOW risk
- MEDIUM risk
- HIGH risk

with operational actions.

### Human-in-the-loop
Simulates expert validation for critical decisions.

### SAP Workflow Simulation
Maps AI decisions into SAP-oriented workflow states.

## Reproducibility

Install dependencies:


pip install -r requirements.txt


Run experiments from:


scripts_and_models/


## Paper

The corresponding IEEE conference paper is available in:


latex/main.tex