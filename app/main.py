from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from .model import predict_batch
import pandas as pd
import io



app = FastAPI(title="Heart Attack Risk Prediction API")

app.mount("/static", StaticFiles(directory=r"c:\Users\schwa\Projects\heart attack prediction\app\static"), name="static")

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-csv", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    contents = await file.read()  
    df = pd.read_csv(io.BytesIO(contents))  
    cat_features = [
        'diabetes', 'family_history', 'smoking', 'obesity',
        'alcohol_consumption', 'diet', 'previous_heart_problems',
        'medication_use', 'stress_level', 'physical_activity_days_per_week', 'gender'
    ]
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float).round().astype(int).astype(str)
    ids = df["id"].values  
    X = df.drop(columns=["id", "ck-mb", "troponin"])  
    
    predictions = predict_batch(X)  
    results = list(zip(ids, predictions))
    
    return templates.TemplateResponse("results.html", {"request": request, "results": results})

