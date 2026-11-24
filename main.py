import joblib
import numpy as np 
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field 

app = FastAPI(title="Customer Churn Prediction API",description="api service powered by fastapi and scikit-learn",version="1.0.0")
try:
    model=joblib.load("Churn_model.pkl")
except Exception:
    model=None
class CustomerInput(BaseModel):
    tenure:int=Field(...,ge=1,le=120,description="Months with Company",example=12)
    monthly_charges:float=Field(...,gt=0,description="Monthly Bill",example=98.0)
    total_charges:float=Field(...,gt=0,description="Total Amount",example=3095.0)


class predictionoutput(BaseModel):
    churn_prediction:int
    churn_probability:float
    risk_level:str


@app.get("/")
def home():
    return{"Message":"Welcome to Customer Churn Prediction API"}
@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500,detail="ML Model is Missing")
    return{"Status":"Healthy","Model_loaded":True}

@app.post("/predict",response_model=predictionoutput)
def predict_churn(data:CustomerInput):
    if model is None:
        raise HTTPException(status_code=500,detail="Model not loaded")
    features=np.array([[data.tenure,data.monthly_charges,data.total_charges]])

    pred_class=int(model.predict(features)[0])
    pred_prob=float(model.predict_proba(features)[0][1])

    risk="High" if pred_class==1 else "Low risk"

    return predictionoutput(churn_prediction=pred_class,churn_probability=pred_prob,risk_level=risk)
    

