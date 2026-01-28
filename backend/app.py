from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="Customer Churn Prediction API")

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load model + feature names
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features_names"]

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

@app.get("/")
def home():
    return {"status": "Customer Churn API is running 🚀"}
@app.post("/predict")
def predict(data: CustomerData):

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Apply encoders safely
    for col, encoder in encoders.items():
        if col in df.columns:
            val = df[col].iloc[0]

            # Handle unseen categories safely
            if val not in encoder.classes_:
                val = encoder.classes_[0]

            df[col] = encoder.transform([val])

    # Add missing features (VERY IMPORTANT)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns exactly as training
    df = df[feature_names]

    # Ensure numeric
    df = df.astype(float)

    # Predict
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]

    return {
        "churn": "Yes" if int(pred) == 1 else "No",
        "confidence": round(float(max(prob)) * 100, 2)
    }

