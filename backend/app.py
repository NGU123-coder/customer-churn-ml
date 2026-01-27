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

    # Convert input JSON to DataFrame
    df = pd.DataFrame([data.dict()])

    # Apply label encoders safely
    for column, encoder in encoders.items():
        if column in df.columns:
            value = df[column].iloc[0]

            if value in encoder.classes_:
                df[column] = encoder.transform([value])
            else:
                # fallback for unseen category
                df[column] = encoder.transform([encoder.classes_[0]])

    # Ensure feature order matches training
    df = df[feature_names]

    # XGBoost expects numeric float input
    df = df.astype(float)

    # Make prediction
    prediction = model.predict(df)
    probability = model.predict_proba(df)

    return {
        "churn": "Yes" if int(prediction[0]) == 1 else "No",
        "confidence": round(float(max(probability[0])) * 100, 2)
    }
