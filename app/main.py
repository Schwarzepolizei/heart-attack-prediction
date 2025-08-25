from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from .schemas import PatientData
from .model import predict_single, predict_batch
import pandas as pd


app = FastAPI(title="Heart Attack Risk Prediction API")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-csv", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    ids = df["id"].values  
    X = df.drop(columns=["id"])  
    
    predictions = predict_batch(X)  
    
    results = list(zip(ids, predictions))
    
    return templates.TemplateResponse("results.html", {"request": request, "results": results})


@app.post("/predict")
def predict_risk(patient: dict):
    patient_id = patient["id"]
    patient_data = {k:v for k,v in patient.items() if k != "id"}  
    
    prediction = predict_single(patient_data)
    return {"id": patient_id, "prediction": prediction}

