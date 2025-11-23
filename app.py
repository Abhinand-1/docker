from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import List

class PredictRequest(BaseModel):
    text: str

class BatchPredictRequest(BaseModel):
    texts: List[str]

app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message":"News Sentiment Classifier API"}

@app.post("/predict")
def predict(req: PredictRequest):
    pred = model.predict([req.text])[0]
    return {"sentiment": pred}

@app.post("/batch_predict")
def batch(req: BatchPredictRequest):
    preds = model.predict(req.texts).tolist()
    return {"results": preds}
