from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

app = FastAPI()

# Charger le modèle entraîné
with open("models/model_cut_prediction.pkl", "rb") as f:
    model = pickle.load(f)

class DiamondFeatures(BaseModel):
    carat: float
    depth: float
    table: float
    x: float
    y: float
    z: float
    color: int
    clarity: int

@app.get("/")
def read_root():
    return {"message": "API Diamond Cut Prediction ready!"}

@app.post("/predict")
def predict(features: DiamondFeatures):
    input_array = np.array([[features.carat, features.depth, features.table,
                             features.x, features.y, features.z,
                             features.color, features.clarity]])
    prediction = model.predict(input_array)
    return {"predicted_cut": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

