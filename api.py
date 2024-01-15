import pandas as pd
import torch
from fastapi import FastAPI

from typing import List
from pydantic import BaseModel

from text2emoji.data.text_processing import preprocess_text
from text2emoji.data.embedding_generation import make_w2v_embeddings, make_mobert_embeddings
from text2emoji.models.nn_classifier import get_probabilities
from text2emoji.models.unfrozen_transformer import get_all_class_probabilities

from create_embedding import load_embedding_model

EMBEDDING_TYPE = "unfrozen_bert"


class EmojiPrediction(BaseModel):
    label: int
    emoji: str
    name: str
    probability: float


class QueryResponse(BaseModel):
    query: str
    results: List[EmojiPrediction]


app = FastAPI()

# Load the model and tokenizer. We can ignore the v_size as we don't need it for prediction.
model, _, tokenizer = load_embedding_model(EMBEDDING_TYPE)
device = "cpu"
model.to(device)

# Classifier layer to be used for prediction
# The layers differ for each embedding type
# Not used for unfrozen_bert
if EMBEDDING_TYPE != "unfrozen_bert":
    classifier = torch.load(f'out/{EMBEDDING_TYPE}/best_model.pt', map_location=device)


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

    return QueryResponse(
        query=text,
        results=labels.to_dict(orient="records")
    )


def generate_probabilities_classifier(cleaned_text):
    """
    Generate probabilities for the given text using the classifier layer.
    These models are frozen and only the classifier layer is trained.

    Args:
        cleaned_text (string): The text to generate probabilities for.

    Returns:
        list: List of probabilities for each emoji.
    """

    # Convert into a pandas dataframe with one row and the text column
    df = pd.DataFrame({'text': [cleaned_text]})

    # Generate embeddings for the text
    name = "production"
    if EMBEDDING_TYPE == "word2vec":
        embeddings, _, _ = make_w2v_embeddings(df, name, model)
    elif EMBEDDING_TYPE == "mobert":
        embeddings = make_mobert_embeddings(df, name, tokenizer, model)

    embeddings = torch.tensor(embeddings).float().to(device)
    probabilities = get_probabilities(classifier, embeddings)

    return probabilities[0]


def generate_probabilities_transformer(cleaned_text):
    """
    Generate probabilities for the given text using the transformer model.

    Args:
        cleaned_text (string): The text to generate probabilities for.

    Returns:
        list: List of probabilities for each emoji.
    """

    predictions = get_all_class_probabilities(cleaned_text, model, tokenizer)

    return predictions


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

    cleaned_text = preprocess_text(text, EMBEDDING_TYPE)

    if EMBEDDING_TYPE == "unfrozen_bert":
        return generate_probabilities_transformer(cleaned_text)
    else:
        return generate_probabilities_classifier(cleaned_text)


if __name__ == "__main__":
    text = input("Enter text: ")
    output = get_emoji_probs(text)
    print(output)
