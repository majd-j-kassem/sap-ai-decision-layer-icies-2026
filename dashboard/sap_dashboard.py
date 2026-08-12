import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="SAP Predictive Decision Layer",
    layout="wide",
)

st.title("SAP-Oriented Predictive Decision Dashboard")
st.caption("IIoT/ML Decision Layer → SAP-Oriented Workflow Simulation")

st.sidebar.header("Order Input")

shipping_mode = st.sidebar.selectbox(
    "Shipping Mode",
    ["Standard Class", "First Class", "Second Class", "Same Day"],
)

market = st.sidebar.selectbox(
    "Market",
    ["USCA", "Europe", "LATAM", "Pacific Asia", "Africa"],
)

order_region = st.sidebar.text_input(
    "Order Region",
    "West of USA",
)

quantity = st.sidebar.number_input(
    "Order Item Quantity",
    min_value=1,
    value=2,
)

price = st.sidebar.number_input(
    "Product Price",
    min_value=0.0,
    value=59.99,
)

late_delivery_cost = st.sidebar.number_input(
    "Late Delivery Cost",
    min_value=0.0,
    value=100.0,
)

intervention_cost = st.sidebar.number_input(
    "Intervention Cost",
    min_value=0.0,
    value=10.0,
)

if st.button("Analyze Order", type="primary"):

    payload = {
        "shipping_mode": shipping_mode,
        "market": market,
        "order_region": order_region,
        "order_item_quantity": quantity,
        "product_price": price,
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )

        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"]
        decision = result["decision"]

        probability = prediction["late_delivery_probability"]
        risk = prediction["risk_level"]
        action = decision["recommended_action"]
        human_review = decision["human_review_required"]

        baseline_cost = probability * late_delivery_cost

        if risk == "HIGH":
            residual_probability = probability * 0.50
        elif risk == "MEDIUM":
            residual_probability = probability * 0.75
        else:
            residual_probability = probability

        optimized_cost = (
            residual_probability * late_delivery_cost
            + (intervention_cost if human_review else 0)
        )

        economic_impact = baseline_cost - optimized_cost

        st.subheader("Prediction")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Late Delivery Probability",
            f"{probability:.2%}",
        )

        col2.metric(
            "Risk Level",
            risk,
        )

        col3.metric(
            "Human Review",
            "Required" if human_review else "Not Required",
        )

        st.divider()

        st.subheader("SAP-Oriented Decision")

        col1, col2 = st.columns(2)

        col1.info(
            f"Recommended SAP Action\n\n"
            f"**{action}**"
        )

        if human_review:
            col2.warning(
                "Human-in-the-loop intervention required."
            )
        else:
            col2.success(
                "Order can proceed automatically."
            )

        st.divider()

        st.subheader("Economic Impact")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Baseline Expected Cost",
            f"${baseline_cost:,.2f}",
        )

        col2.metric(
            "Decision Cost",
            f"${optimized_cost:,.2f}",
        )

        if economic_impact >= 0:
            col3.metric(
                "Expected Saving",
                f"${economic_impact:,.2f}",
            )
        else:
            col3.metric(
                "Expected Loss",
                f"${economic_impact:,.2f}",
            )

        st.divider()

        st.subheader("Decision Explanation")

        for reason in result.get("explanation", []):
            st.write(f"- {reason}")

        st.subheader("Integration Status")

        st.json(
            result.get(
                "sap_integration",
                {},
            )
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Decision API is not running. "
            "Start 15_decision_api.py first."
        )

    except requests.exceptions.RequestException as exc:
        st.error(f"API error: {exc}")