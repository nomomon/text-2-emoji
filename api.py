import pandas as pd
import gensim.downloader as w2v_api
import torch
from fastapi import FastAPI

from typing import List
from pydantic import BaseModel

from text2emoji.data.text_processing import preprocess_text
from text2emoji.data.embedding_generation import make_w2v_embeddings
from text2emoji.models.nn_classifier import get_probabilities


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

    probs = get_emoji_probs(text)

    labels = pd.read_csv('./data/bronze/mapping.csv')
    labels["probability"] = probs
    labels = labels.sort_values(by="probability", ascending=False)

    return QueryResponse(results=labels.to_dict(orient="records"))


def get_emoji_probs(text: str):
    """
    Returns a list of probabilities for each emoji.

    Args:
        text (str): The text to predict emoji for.

    Returns:
        list: The list of probabilities for each emoji.
    """

    cleaned_text = preprocess_text(text)

    # convert into a pandas dataframe with one row and the text column
    df = pd.DataFrame({'text': [cleaned_text]})

    # We should ideally load this before hand
    w2v_model = w2v_api.load("word2vec-google-news-300")
    embeddings, _, _ = make_w2v_embeddings(df, "Production", w2v_model)

    classifier = torch.load('out/best_model.pt')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier.to(device)

    embeddings = torch.tensor(embeddings).float().to(device)
    probabilites = get_probabilities(classifier, embeddings)

    return probabilites[0]


if __name__ == "__main__":
    text = input("Enter text: ")
    output = get_emoji_probs(text)
