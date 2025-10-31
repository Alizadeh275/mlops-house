from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("ml/model.pkl")


class PredictRequest(BaseModel):
    features: list  # list of floats for input features


@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array(req.features).reshape(1, -1)
    y_pred = model.predict(x)
    return {"prediction": y_pred.tolist()}
