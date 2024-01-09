import numpy as np
import pandas as pd
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/get_emoji")
async def get_emoji(text: str):
    probs = get_emoji(text)

    labels = pd.read_csv('./data/bronze/mapping.csv')
    labels["probability"] = probs

    return labels.to_dict(orient='records')

def get_emoji(text: str):
    a = np.random.rand(20)
    return a / a.sum()