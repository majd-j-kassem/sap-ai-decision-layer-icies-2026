# An Explainable AI Decision Layer for Human-in-the-loop SAP ERP Workflow Integration in Supply Chain Operations

[![Conference](https://shields.io)](http://icies-conference.org)
[![Python](https://shields.io)](https://python.org)
[![Framework](https://shields.io)](https://streamlit.io)

## 📌 Repository Status

This repository contains the official research prototype developed for the **ICIES 2026** conference. 

The `main` branch includes the comprehensive replication package:
* 🧪 **Experimental Scripts:** Full data pipelines and statistical tests.
* 🤖 **Trained Models:** Serialized machine learning artifacts.
* 📊 **Evaluation Results:** Raw outputs, calibration curves, and figures.
* ⚙️ **SAP-Oriented Workflow:** Simulation engine for enterprise integration.
* 🖥️ **Dashboard Prototype:** Interactive Streamlit interface.
* 📝 **LaTeX Source:** Complete academic paper source files.

> ⚠️ **Note:** This repository is intended strictly for research reproducibility and prototype demonstration. The implementation represents an experimental framework and not a production-grade SAP deployment.

---

## 🔍 Overview

This framework bridges the gap between predictive machine learning and operational enterprise workflows. It introduces an **Explainable AI (XAI) decision layer** for supply chain risk management, connecting raw delivery risk probabilities to structured, cost-sensitive downstream actions.

### Core Capabilities:
* **Risk Stratification:** Classifies delivery risks into actionable operational states.
* **Human-in-the-Loop (HITL):** Escalation mechanisms for high-uncertainty decisions.
* **Cost-Sensitive Evaluation:** Economic assessment of intervention policies vs. business-as-usual.
* **SAP Workflow Simulation:** Direct mapping of AI explanations to standard SAP ERP status codes.

---

## 📂 Project Structure

```text
2026_ICIES/
│
├── data/
│   └── DataCoSupplyChainDataset.csv  # External dataset (must be downloaded)
│
├── models/
│   ├── lightgbm_model.joblib
│   └── xgboost_model.joblib
│
├── scripts_and_models/
│   ├── 01_data_profile.py
│   ├── 02_relationship_tests.py
│   ├── 03_feature_screening.py
│   ├── 04_model_training.py
│   ├── 09_xgboost_lightgbm.py
│   ├── 12_decision_layer.py
│   ├── 13_sap_integration_prototype.py
│   └── 34_paper_final_results.py
│
├── src/
│   ├── api/                          # FastAPI endpoints for decision routing
│   ├── decision/                     # Logic for risk stratification & cost optimization
│   ├── models/                       # Inference wrappers and explainers
│   └── sap/                          # SAP IDoc and workflow mapping simulation
│
├── dashboard/
│   └── sap_dashboard.py              # Streamlit interactive UI
│
├── results/                          # Generated plots, metrics, and tables
│
├── latex/
│   └── release/                      # Academic paper source files
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Repository documentation
```

---

## 📊 Dataset

The experimental pipeline utilizes the publicly available **DataCo Supply Chain Dataset**. Due to file size limitations and licensing constraints, the dataset is not hosted directly in this repository.

### Setup Instructions:
1. **Download** the raw data from the official source:
   [DataCo Dataset Download Link](https://prod-dcd-datasets-public-files-eu-west-1.amazonaws.com/b60060a2-e731-4745-8d51-3db158a1add7)
2. **Place** the unzipped CSV file exactly at the following path:
   ```bash
   data/DataCoSupplyChainDataset.csv
   ```

*All downstream experimental tables, figures, and evaluation logs can be deterministically regenerated from this file.*

---

---

## 🛠️ Installation & Setup

Follow these steps to clone the repository, set up an isolated virtual environment, and install the required dependencies directly from the project root:

```bash
# 1. Clone the repository from GitHub
git clone https://github.com/majd-j-kassem/sap-ai-decision-layer-icies-2026.git

# 2. Navigate to the project root directory
cd sap-ai-decision-layer-icies-2026

# 3. Create an isolated virtual environment (.venv)
python3 -m venv .venv

# 4. Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (Command Prompt):
.venv\Scripts\activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 5. Upgrade pip and install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 **Tip:** Always ensure that your terminal prompt shows `(.venv)` before executing any experimental scripts or launching the dashboard to guarantee that dependencies are read from the local environment.

---

## 🧪 Running Experiments

The research execution pipeline is modularized into sequential steps. Execute them from the project root directory.

### 1. Data Understanding & Profiling
Generates descriptive dataset statistics, missing-value distributions, and target variable balances.
```bash
python scripts_and_models/01_data_profile.py
```

### 2. Statistical Analysis
Performs categorical relationship tests and Chi-Square (\(\chi^2\)) statistical validations.
```bash
python scripts_and_models/02_relationship_tests.py
```

### 3. Feature Screening
Identifies and filters relevant operational variables used as features in predictive modeling.
```bash
python scripts_and_models/03_feature_screening.py
```

### 4. Model Training & Evaluation
Trains and compares benchmarks (CART, Logistic Regression, Random Forest) against advanced gradient boosters (XGBoost, LightGBM).
```bash
python scripts_and_models/04_model_training.py
python scripts_and_models/09_xgboost_lightgbm.py
```

### 5. Decision Layer & Workflow Simulation
Generates risk states, maps operational policies, triggers HITL escalations, and outputs simulated SAP-compatible logs.
```bash
python scripts_and_models/12_decision_layer.py
python scripts_and_models/13_sap_integration_prototype.py
```

---

## 🚀 Serving the Prototype

### 1. Decision API (FastAPI)
The repository includes an API microservice demonstrating how predictions are served and translated into operational decision states.

```bash
# Activate environment and launch the server
source .venv/bin/activate
uvicorn src.api.decision_api:app --reload
```
* **Interactive Docs URL:** [http://127.0.0](http://127.0.0)

### 2. SAP Decision Dashboard (Streamlit)
To explore the interactive, human-in-the-loop simulation UI:

```bash
source .venv/bin/activate
streamlit run dashboard/sap_dashboard.py
```
* **Local Web URL:** [http://localhost:8501](http://localhost:8501)
* **Features:** Dynamic risk category adjustment, SHAP explanation rendering, and human override logs.

---

## 📈 Paper Reproduction (LaTeX)

To compile the LaTeX source files of the paper, ensure you have a standard distribution installed (e.g., TeX Live, MiKTeX).

```bash
cd latex/release/
latexmk -pdf paper_file.tex
```
The compiled output will be generated as `paper_file.pdf`.

---

## 🔄 Complete Replication Workflow

For an end-to-end exact reproduction of the study results, follow this sequential execution pathway:

```text
[1. Download Dataset] ──> [2. Data Profiling] ──> [3. Feature Screening]
                                                         │
                                                         ▼
[6. Risk Stratification] <── [5. Model Calibration] <── [4. Model Training]
         │
         ▼
[7. Decision Policies] ──> [8. HITL Evaluation] ──> [9. Cost Analysis]
                                                         │
                                                         ▼
                                             [10. Regenerate Paper Plots]
```

---

## 🛑 Research Limitations

* **Simulation Scope:** The SAP integration component is implemented as a high-fidelity workflow simulation mapping variables to logical enterprise pathways. It does **not** feature a live NetWeaver/RFC connection to a production SAP instance.
* **Economic Assumptions:** The cost-benefit matrix and financial saving metrics (e.g., the reported 3.40% operational saving) are evaluated using fixed parameter cost profiles detailed in the paper. True organizational cost reductions will scale dynamically based on individual corporate SLA penalties and mitigation costs.

---

## ✍️ Citation
