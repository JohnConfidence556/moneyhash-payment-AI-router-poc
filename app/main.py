import os
import joblib
import pandas as pd
import logging 
from datetime import datetime 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# MONITORING & LOGGING SETUP
# This creates a file named 'prediction_audit.log' in your container
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("prediction_audit.log"), # Saves to file
        logging.StreamHandler() # Also prints to Render/Docker logs
    ]
)
logger = logging.getLogger("MoneyHashRouter")

# Using relative paths is crucial for Docker and Cloud hosting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "moneyhash_router_v1.pkl")

app = FastAPI(
    title="MoneyHash Predictive Routing API",
    description="ML-powered payment orchestration for high-success routing.",
    version="1.0.0"
)

# Define Request Schema
class PaymentTransaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount must be positive")
    country: str = Field(..., example="Nigeria")
    method: str = Field(..., example="card")
    hour: int = Field(..., ge=0, le=23)
    time_of_day: str = Field(..., example="Afternoon") 
    gateway: str = Field(..., example="Paystack")      

# We load this once at startup to save memory and reduce latency
try:
    predictive_router = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully from app/models/")
except Exception as e:
    logger.error(f"❌ Critical Error: Could not load model. {e}")
    predictive_router = None

@app.get("/health")
def health_check():
    """Verify API is live and model is loaded."""
    if predictive_router:
        return {"status": "healthy", "model_version": "v1.0.0"}
    return {"status": "degraded", "error": "Model not found"}

@app.post("/v1/predict/route")
async def predict_route(transaction: PaymentTransaction):
    if not predictive_router:
        logger.warning("Attempted prediction while model was unavailable.")
        raise HTTPException(status_code=503, detail="Model is currently unavailable")

    try:
        # Convert Pydantic object to DataFrame for the Scikit-learn Pipeline
        input_data = transaction.dict()
        input_df = pd.DataFrame([input_data])
        
        # Binary prediction (1 = Success, 0 = Failure)
        prediction = int(predictive_router.predict(input_df)[0])
        
        # Confidence score
        probabilities = predictive_router.predict_proba(input_df)[0]
        confidence = float(max(probabilities))

        recommendation = "PROCEED" if prediction == 1 else "TRIGGER_FALLBACK"

        # AUDIT LOGGING
        # We log the input and output so we can monitor performance
        log_payload = {
            "event": "PREDICTION_MADE",
            "transaction": input_data,
            "result": recommendation,
            "confidence": round(confidence, 4)
        }
        logger.info(f"AUDIT: {log_payload}")

        return {
            "prediction_status": "SUCCESS" if prediction == 1 else "FAILURE_RISK",
            "confidence_score": round(confidence, 4),
            "recommendation": recommendation,
            "provider_hint": "Optimized by XGBoost for current hour/amount"
        }
    except Exception as e:
        logger.error(f"Inference error for transaction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)