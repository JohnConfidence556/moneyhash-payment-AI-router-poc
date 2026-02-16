import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="MoneyHash AI Router POC", layout="wide")

st.title("🛡️ MoneyHash Routing Simulator: Rules vs. AI")
st.markdown("Comparing the **25.8% Baseline** against the **Dockerized XGBoost Model**.")

# Transaction Inputs
with st.sidebar:
    st.header("Transaction Setup")
    amt = st.number_input("Transaction Amount ($)", value=5000.0, step=100.0)
    geo = st.selectbox("Destination Country", ["Nigeria", "Egypt", "Kenya", "South Africa"])
    pay = st.selectbox("Payment Method", ["card", "bank_transfer", "mobile_money"])
    gtway = st.selectbox("Target Gateway", ["Paystack", "Stripe", "Fawry", "Flutterwave"])
    hr = st.slider("Hour of Transaction (0-23)", 0, 23, 14)
    tod = st.selectbox("Time Category", ["Morning", "Afternoon", "Evening", "Night"])

# Side-by-Side Duel
col1, col2 = st.columns(2)

with col1:
    st.subheader("📜 Rule-Based (Baseline)")
    # Simulating the static rules i used for the baseline project phase
    if amt > 10000 and geo == "Nigeria":
        st.error("ACTION: BLOCK (Static Rule: Limit Exceeded)")
        st.caption("Result: Potential Revenue Loss")
    elif hr > 22:
        st.warning("ACTION: DELAY (Static Rule: Maintenance window)")
    else:
        st.success("ACTION: PROCEED (Standard Route)")

with col2:
    st.subheader("🤖 ML-Optimized (Docker API)")
    if st.button("Query AI Intelligence"):
        # Matching the 6 fields from my Pydantic Schema exactly
        payload = {
            "amount": amt,
            "country": geo,
            "method": pay,
            "hour": hr,
            "time_of_day": tod,
            "gateway": gtway
        }
        
        try:
            
            response = requests.post("https://moneyhash-payment-ai-router-poc.onrender.com/v1/predict/route", json=payload)
            
            if response.status_code == 200:
                res = response.json()
                if res["prediction_status"] == "SUCCESS":
                    st.success(f"RECOMMENDATION: {res['recommendation']}")
                    st.metric("Model Confidence", res['confidence_score'])
                else:
                    st.error(f"RISK DETECTED: {res['recommendation']}")
                    st.metric("Model Confidence", res['confidence_score'])
                
                st.info(f"💡 AI Insight: {res['provider_hint']}")
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Connection Failed: Is the Docker container running? (Error: {e})")

# Explainability Note
st.divider()
st.markdown("### 🔍 The 'Why' (SHAP Intelligence)")
st.write("The ML model utilizes SHAP values to understand feature importance, ensuring it doesn't just block transactions based on a single rule (like 'Amount'), but looks at the full context of country, time, and method.")