from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np

app = FastAPI(title="Customer Churn Prediction API")

# ---------------- LOAD ARTIFACTS ---------------- #

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features_names"]

# ---------------- INPUT SCHEMA ---------------- #

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ---------------- ROUTES ---------------- #

@app.get("/")
def home():
    return {"status": "Customer Churn API is running 🚀"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Apply encoders
        for col, encoder in encoders.items():
            if col in df.columns:
                val = df.at[0, col]
                if val in encoder.classes_:
                    df[col] = encoder.transform([val])
                else:
                    df[col] = encoder.transform([encoder.classes_[0]])

        # Add missing columns (VERY IMPORTANT)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # Keep only trained features in correct order
        df = df[feature_names]

        # Ensure numeric
        df = df.astype(float)

        # Predict
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0].max()

        return {
            "churn": "Yes" if int(pred) == 1 else "No",
            "confidence": round(float(prob) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
