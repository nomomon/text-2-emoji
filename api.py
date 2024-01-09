import numpy as np
import pandas as pd
from fastapi import FastAPI

from typing import List
from pydantic import BaseModel

class EmojiPrediction(BaseModel):
    label: int
    emoji: str
    name: str
    probability: float

class QueryResponse(BaseModel):
    results: List[EmojiPrediction]

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/get_emoji")
async def get_emoji(text: str) -> QueryResponse:
    """
    Returns a list of emoji predictions for the given text. The list is sorted by probability. And the probability sum up to 1.
    """

    probs = get_emoji(text)

    labels = pd.read_csv('./data/bronze/mapping.csv')
    labels["probability"] = probs
    labels = labels.sort_values(by="probability", ascending=False)

    return QueryResponse(results=labels.to_dict(orient="records"))

def get_emoji(text: str):
    a = np.random.rand(20)
    return a / a.sum()
