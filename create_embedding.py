import os
import gensim.downloader as w2v_api
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import torch

from text2emoji.data.embedding_generation import make_w2v_embeddings, make_mobert_embeddings


def load_embedding_model(encoder_type, device=None):
    """
    Load the model and tokenizer for the given encoder type.

    Args:
        encoder_type (str): The encoder type to load.

    Raises:
        ValueError: Raised if an invalid encoder type is given.

    Returns:
        model: A word2vec or transformer model.
        v_size: The size of the vector space.
        tokenizer: A tokenizer for the transformer model.
    """

    tokenizer = None
    if encoder_type == "word2vec":
        model = w2v_api.load("word2vec-google-news-300")
        v_size = model.vector_size
    elif encoder_type == "mobert":
        tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
        model = AutoModel.from_pretrained('google/mobilebert-uncased')
        v_size = 512
    elif encoder_type == 'unfrozen_bert':
        if not os.path.exists('out/unfrozen_bert/best_model.pt'):
            print("\n\nUnfrozen BERT model not found. Starting download...\n\n")
            os.system("gdown --id 1Zi6GP6DC_iLk_MtIVuXugR57PUS2Oowj -O out/unfrozen_bert/best_model.pt")

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = torch.load('out/unfrozen_bert/best_model.pt', map_location=device)
        v_size = 768
    else:
        raise ValueError("Invalid encoder type")

    return model, v_size, tokenizer


def make_sentence_embeddings(encoder_type="word2vec"):
    """
    Create sentence embeddings for the given dataframe using word2vec or mobert.

    Args:
        encoder_type (str, optional):
        The technique to use for embedding generation. Defaults to "word2vec".

    Raises:
        ValueError: An invalid encoder type was given.
    """

    if encoder_type == 'transformer':
        print("Skipping embedding generation for transformer")
        return

    train = pd.read_csv('./data/silver/train.csv')
    valid = pd.read_csv('./data/silver/valid.csv')
    test = pd.read_csv('./data/silver/test.csv')

    model, v_size, tokenizer = load_embedding_model(encoder_type)

    n_invalid_texts = 0
    n_only_unknown_words = 0

    for df, name in [(train, "train"), (valid, "valid"), (test, "test")]:

        target = df.label.values

        if encoder_type == "word2vec":
            features, set_invalid_texts, set_unknown_words = make_w2v_embeddings(df, name, model)

            n_invalid_texts += set_invalid_texts
            n_only_unknown_words += set_unknown_words

        elif encoder_type == "mobert":
            features = make_mobert_embeddings(df, name, tokenizer, model)

        # Normalize the features to unit length
        # Seems like this results in worse performance
        features = normalize(features)

        assert features.shape == (len(df), v_size)
        assert len(features) == len(target)

        np.save(f'./data/gold/{name}_{encoder_type}_features.npy', features)
        np.save(f'./data/gold/{name}_{encoder_type}_target.npy', target)

    # TODO: possibly we can just drop the invalid texts
    #       in earlier preprocessing steps, so we can
    #       replace with an assert here instead (mansur)
    print(f"Number of invalid texts: {n_invalid_texts}")
    print(f"Number of texts with only unknown words: {n_only_unknown_words}")


if __name__ == '__main__':
    make_sentence_embeddings("word2vec")
    print("done")
