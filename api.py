import pandas as pd
import torch
from fastapi import FastAPI

from typing import List
from pydantic import BaseModel

from text2emoji.data.text_processing import preprocess_text
from text2emoji.data.embedding_generation import make_w2v_embeddings, make_mobert_embeddings
from text2emoji.models.nn_classifier import get_probabilities
from create_embedding import load_embedding_model

EMBEDDING_TYPE = "word2vec"


class EmojiPrediction(BaseModel):
    label: int
    emoji: str
    name: str
    probability: float


class QueryResponse(BaseModel):
    results: List[EmojiPrediction]


app = FastAPI()

# Load the model and tokenizer. We can ignore the v_size as we don't need it for prediction.
embedding_model, _, tokenizer = load_embedding_model(EMBEDDING_TYPE)

# Classifier layer to be used for prediction
# The layers differ for each embedding type
classifier = torch.load(f'out/{EMBEDDING_TYPE}/best_model.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier.to(device)


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

    # Replace %20 with spaces
    text = text.replace("%20", " ")

    cleaned_text = preprocess_text(text)

    # convert into a pandas dataframe with one row and the text column
    df = pd.DataFrame({'text': [cleaned_text]})

    # Generate embeddings for the text
    name = "production"
    if EMBEDDING_TYPE == "word2vec":
        embeddings, _, _ = make_w2v_embeddings(df, name, embedding_model)

    elif EMBEDDING_TYPE == "mobert":
        embeddings = make_mobert_embeddings(df, name, tokenizer, embedding_model)

    embeddings = torch.tensor(embeddings).float().to(device)
    probabilites = get_probabilities(classifier, embeddings)

    return probabilites[0]


if __name__ == "__main__":
    text = input("Enter text: ")
    output = get_emoji_probs(text)
    print(output)
