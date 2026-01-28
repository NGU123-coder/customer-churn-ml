from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="Customer Churn Prediction API")

# ---------------- LOAD FILES ----------------

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features_names"]

# ---------------- INPUT SCHEMA ----------------

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

# ---------------- ROUTES ----------------

@app.get("/")
def home():
    return {"status": "Customer Churn API is running 🚀"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert request to DataFrame WITH column names
        df = pd.DataFrame([data.model_dump()])

        # Apply encoders
        for col, encoder in encoders.items():
            if col in df.columns:
                value = df[col].iloc[0]
                if value in encoder.classes_:
                    df[col] = encoder.transform([value])
                else:
                    df[col] = encoder.transform([encoder.classes_[0]])

        # 🔴 CRITICAL LINE (THIS FIXES YOUR ERROR)
        df = df.reindex(columns=feature_names)

        # Convert to numeric
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
