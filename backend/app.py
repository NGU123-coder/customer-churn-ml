from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="Customer Churn Prediction API")

# =========================
# LOAD MODEL & ENCODERS
# =========================

with open("customer_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# =========================
# FEATURE ORDER (MUST MATCH TRAINING)
# =========================

FEATURE_ORDER = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

# =========================
# INPUT SCHEMA
# =========================

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

# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {"status": "Customer Churn API is running 🚀"}

@app.post("/predict")
def predict(data: CustomerData):

    input_dict = data.dict()

    # Apply encoders
    for column, encoder in encoders.items():
        value = input_dict[column]

        if value in encoder.classes_:
            input_dict[column] = encoder.transform([value])[0]
        else:
            # fallback for unseen category
            input_dict[column] = encoder.transform([encoder.classes_[0]])[0]

    # Create DataFrame in correct order
    df = pd.DataFrame(
        [[input_dict[col] for col in FEATURE_ORDER]],
        columns=FEATURE_ORDER
    )

    # Ensure numeric dtype
    df = df.astype(float)

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": "Yes" if int(prediction) == 1 else "No",
        "churn_probability": round(float(probability), 3)
    }
