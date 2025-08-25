from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from .schemas import PatientData
from .model import predict_single, predict_batch
import pandas as pd


app = FastAPI(title="Heart Attack Risk Prediction API")

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-csv", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    predictions = predict_batch(df)
    return templates.TemplateResponse("results.html", {"request": request, "predictions": predictions})

@app.post("/predict")
def predict_risk(patient: PatientData):
    prediction = predict_single(patient.dict())
    return {"prediction": prediction}
