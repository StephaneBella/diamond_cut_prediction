from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd 
import os

app = FastAPI()

# Chemins absolus
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "model_cut_prediction.pkl")
encoder_path = os.path.join(BASE_DIR, "..", "models", "target_encoder.pkl")

# Charger le modèle entraîné
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Charger l'encodeur de la target
with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

class DiamondFeatures(BaseModel):
    carat: float
    depth: float
    table: float
    x: float
    y: float
    z: float
    color: str
    clarity: str

@app.get("/")
def read_root():
    return {"message": "API Diamond Cut Prediction ready!"}
 

@app.post("/predict")
def predict(features: DiamondFeatures):
    # Crée un DataFrame avec les colonnes dans le bon ordre
    input_df = pd.DataFrame([{
        "carat": features.carat,
        "depth": features.depth,
        "table": features.table,
        "x": features.x,
        "y": features.y,
        "z": features.z,
        "color": features.color,
        "clarity": features.clarity
    }])

    # Prédiction
    prediction = model.predict(input_df)

    # Inverse transform
    decoded = encoder.inverse_transform(prediction)

    return {"predicted_cut": decoded[0]}



