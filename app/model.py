import pickle
import pandas as pd


with open("cat_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_single(patient_dict):
    df = pd.DataFrame([patient_dict])
    return int(model.predict(df)[0])

def predict_batch(df: pd.DataFrame):
    return model.predict(df)
