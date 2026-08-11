import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "models/lightgbm_model.joblib"

app = FastAPI(
    title="SAP-Oriented Predictive Decision API",
    version="1.0.0",
)


class OrderRequest(BaseModel):
    shipping_mode: str
    market: str
    order_region: str
    order_item_quantity: float
    product_price: float


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


def classify_risk(score):
    if score >= 0.70:
        return "HIGH"
    elif score >= 0.30:
        return "MEDIUM"
    return "LOW"


def recommend_action(risk_level):
    actions = {
        "LOW": "SAP_STANDARD_PROCESS",
        "MEDIUM": "SAP_SHIPPING_PLAN_REVIEW",
        "HIGH": "SAP_ORDER_PRIORITY_REVIEW",
    }

    return actions[risk_level]


def generate_explanation(request, risk_level):
    reasons = []

    if request.shipping_mode in ["First Class", "Second Class"]:
        reasons.append(
            "Shipping mode contributes significantly to predicted risk."
        )

    if request.product_price > 100:
        reasons.append(
            "Product price is relatively high."
        )

    if request.order_item_quantity >= 3:
        reasons.append(
            "Order item quantity is relatively high."
        )

    if not reasons:
        reasons.append(
            "Risk is primarily determined by the learned predictive model."
        )

    return reasons


@app.get("/health")
def health():
    return {
        "status": "OK",
        "service": "SAP Predictive Decision Layer",
    }


@app.post("/predict")
def predict(request: OrderRequest):

    try:
        data = pd.DataFrame(
            [{
                "Shipping Mode": request.shipping_mode,
                "Market": request.market,
                "Order Region": request.order_region,
                "Order Item Quantity": request.order_item_quantity,
                "Product Price": request.product_price,
            }]
        )

        probability = float(
            model.predict_proba(data)[0][1]
        )

        risk_level = classify_risk(probability)
        action = recommend_action(risk_level)
        explanation = generate_explanation(
            request,
            risk_level,
        )

        return {
            "prediction": {
                "late_delivery_probability": round(
                    probability, 4
                ),
                "risk_level": risk_level,
            },
            "decision": {
                "recommended_action": action,
                "human_review_required": (
                    risk_level != "LOW"
                ),
            },
            "explanation": explanation,
            "sap_integration": {
                "status": "READY",
                "target": "SAP Decision / Workflow Layer",
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)