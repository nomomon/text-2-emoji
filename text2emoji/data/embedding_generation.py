import nltk
import numpy as np
from tqdm import tqdm
import torch
import gc


def invalid_text_vec(v_size):
    return np.zeros(v_size, dtype=np.float32)


def make_w2v_embeddings(df, name, w2v_model):

    features = []
    v_size = w2v_model.vector_size
    n_invalid_texts = 0
    n_only_unknown_words = 0

    for row in tqdm(df.itertuples(), total=len(df), desc=name):

        try:
            sentence = nltk.word_tokenize(row.text)
        except TypeError:
            # (e.g. empty sentence)
            n_invalid_texts += 1
            features.append(invalid_text_vec(v_size))
            continue

        word_vectors = []
        for word in sentence:
            try:
                word_vectors.append(w2v_model[word])
            except KeyError:
                # ignore (e.g. unknown word)
                continue

        if len(word_vectors) == 0:
            n_only_unknown_words += 1
            features.append(invalid_text_vec(v_size))
            continue

        word_vectors = np.array(word_vectors)
        mean_vector = word_vectors.mean(axis=0)
        features.append(mean_vector)

    return np.array(features), n_invalid_texts, n_only_unknown_words


def get_tweet_embeddings_mobert(tweets, tokenizer, model):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Add cls and sep tokens
    tweets = ['[CLS] ' + tweet + ' [SEP]' for tweet in tweets]

    # Tokenize all tweets and get attention masks
    encoded_inputs = tokenizer.batch_encode_plus(
        tweets,
        padding='longest',  # Pad sequences to the length of the longest sequence
        truncation=True,  # Truncate sequences if they are longer than the model's max length
        max_length=64,  # Maximum length for the sequences
        return_tensors='pt'  # Return PyTorch tensors
    )

    # Get token IDs and attention masks
    token_ids_tensor = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    # Get embeddings of tweets
    tweet_embeddings = model(token_ids_tensor, attention_mask=attention_mask)

    del token_ids_tensor
    del attention_mask
    gc.collect()

    # Get last hidden states
    return tweet_embeddings.pooler_output.detach().cpu()


def make_mobert_embeddings(df, name, tokenizer, model):

    tweets = df.text.values.tolist()

    embeddings = []
    BATCH_SIZE = 256
    for index in tqdm(range(0, len(tweets), BATCH_SIZE), desc=name):
        embeddings.append(get_tweet_embeddings_mobert(tweets[index:index+BATCH_SIZE], tokenizer, model))

    return torch.cat(embeddings).numpy()
