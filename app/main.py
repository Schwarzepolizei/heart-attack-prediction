from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Загружаем модель
with open("cat_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Heart Attack Risk Prediction API")

class PatientData(BaseModel):
    bmi: float
    sedentary_hours_per_day: float
    cholesterol: float
    triglycerides: float
    systolic_blood_pressure: float
    diastolic_blood_pressure: float
    exercise_hours_per_week: float
    income: float
    heart_rate: float
    age: float
    diet: int
    physical_activity_days_per_week: int
    sleep_hours_per_day: float
    stress_level: int
    blood_sugar: float
    obesity: int
    alcohol_consumption: int
    family_history: int
    previous_heart_problems: int
    diabetes: int
    medication_use: int
    gender: int

@app.post("/predict")
def predict_risk(patient: PatientData):
    input_df = pd.DataFrame([patient.dict()])
    
   
    prediction = model.predict(input_df)[0]
    
    return {"prediction": int(prediction)}