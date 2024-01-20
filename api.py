import pandas as pd
import torch
from fastapi import FastAPI, Query, HTTPException

from typing import List
from pydantic import BaseModel

from text2emoji.data.text_processing import preprocess_text
from text2emoji.data.embedding_generation import make_w2v_embeddings, make_mobert_embeddings
from text2emoji.models.nn_classifier import get_probabilities
from text2emoji.models.unfrozen_transformer import get_all_class_probabilities

from create_embedding import load_embedding_model

EMBEDDING_TYPES = ["word2vec", "mobert", "unfrozen_bert"]


class EmojiPrediction(BaseModel):
    label: int
    emoji: str
    name: str
    probability: float


class QueryResponse(BaseModel):
    query: str
    embedding_type: str = Query(enum=EMBEDDING_TYPES)
    results: List[EmojiPrediction]


app = FastAPI(
    title="Text2Emoji API",
    description="""

API for predicting emoji for a given text. There are three different embedding types available:
- `word2vec`: Uses word2vec embeddings and a simple classifier layer.
- `mobert`: Uses multilingual BERT embeddings and a simple classifier layer.
- `unfrozen_bert`: Uses multilingual BERT embeddings and a transformer model.

There are a total of 20 emojis that can be predicted, which are the following:
    ❤ 😍 😂 💕 🔥  😊 😎 ✨ 💙 😘 
    📷 🇺🇸 ☀ 💜 😉 💯 😁 🎄 📸 😜

## Model Usage

The API can be used by sending a GET request to the `/get_emoji` endpoint. (see below for more details)

## Errors

The API returns a 400 error if the `embedding_type` parameter is invalid.
    """,
    version="0.1.0",
)


device = "cpu"

# Load the models and tokenizers
word2vec_model, _, word2vec_tokenizer = load_embedding_model('word2vec')
word2vec_classifier = torch.load(f'out/word2vec/best_model.pt', map_location=device)

mobert_model, _, mobert_tokenizer = load_embedding_model('mobert')
mobert_model.to(device)
mobert_classifier = torch.load(f'out/mobert/best_model.pt', map_location=device)

unforzen_bert_model, _, unforzen_bert_tokenizer = load_embedding_model('unfrozen_bert', device)
unforzen_bert_model.to(device)


@app.get("/")
async def root():
    """
    Returns a status message, indicating that the API is running.
    """
    return {"status": "ok"}


@app.get("/get_emoji")
async def get_emoji(text: str, embedding_type: str = Query(enum=EMBEDDING_TYPES)) -> QueryResponse:
    """
    Returns a list of emoji predictions for the given text. 
    The list is sorted by probability. 
    And the probabilities sum up to 1.

    Args:  
    - text (str): The text to predict emoji for.   
    - embedding_type (str): The embedding type to use. Must be one of ['word2vec', 'mobert', 'unfrozen_bert'].
    """

    if embedding_type not in EMBEDDING_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid value for embedding_type, must be one of {EMBEDDING_TYPES}, but was {embedding_type}")

    probs = get_emoji_probs(text, embedding_type)

    labels = pd.read_csv('./data/bronze/mapping.csv')
    labels["probability"] = probs
    labels = labels.sort_values(by="probability", ascending=False)

    return QueryResponse(
        query=text,
        embedding_type=embedding_type,
        results=labels.to_dict(orient="records")
    )


def decode_URI(s):
    """
    Decode the given string.
    """
    return s.replace("%20", " ")


def generate_probabilities_classifier(cleaned_text, embedding_type):
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
    if embedding_type == "word2vec":
        embeddings, _, _ = make_w2v_embeddings(df, name, word2vec_model)
        embeddings = torch.tensor(embeddings).float().to(device)
        probabilities = get_probabilities(word2vec_classifier, embeddings)
    elif embedding_type == "mobert":
        embeddings = make_mobert_embeddings(df, name, mobert_tokenizer, mobert_model)
        embeddings = torch.tensor(embeddings).float().to(device)
        probabilities = get_probabilities(mobert_classifier, embeddings)

    return probabilities[0]


def generate_probabilities_transformer(cleaned_text):
    """
    Generate probabilities for the given text using the transformer model.

    Args:
        cleaned_text (string): The text to generate probabilities for.

    Returns:
        list: List of probabilities for each emoji.
    """

    predictions = get_all_class_probabilities(
        cleaned_text, 
        model=unforzen_bert_model,
        tokenizer=unforzen_bert_tokenizer,
    )

    return predictions


def get_emoji_probs(text: str, embedding_type: str):
    """
    Returns a list of probabilities for each emoji.

    Args:
        text (str): The text to predict emoji for.

    Returns:
        list: The list of probabilities for each emoji.
    """

    text = decode_URI(text)

    cleaned_text = preprocess_text(text, embedding_type)

    if embedding_type == "unfrozen_bert":
        return generate_probabilities_transformer(cleaned_text)
    elif embedding_type in ["word2vec", "mobert"]:
        return generate_probabilities_classifier(cleaned_text, embedding_type)
    else:
        raise ValueError(f"Invalid value for embedding_type, must be one of {EMBEDDING_TYPES}")

if __name__ == "__main__":
    text = input("Enter text: ")
    output = get_emoji_probs(text)
    print(output)
