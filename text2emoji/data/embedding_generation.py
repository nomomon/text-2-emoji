import nltk
import numpy as np
from tqdm import tqdm
import torch
import gc


def invalid_text_vec(v_size):
    """
    Create an invalid text vector.

    Args:
        v_size (int): The size of the vector.

    Returns:
        A numpy array of zeros.
    """

    return np.zeros(v_size, dtype=np.float32)


def calculate_idf_scores(df):
    """
    Get the IDF scores of the words in the given dataframe.
    Only used with training data.

    Args:
        df (Dataframe): The dataframe which contains all the tweets.
    """

    # Tokenize each row and store the tokens in a list
    tokenized_texts = [set(nltk.word_tokenize(row.text)) for row in tqdm(df.itertuples(), total=len(df), desc="Tokenizing texts")]

    # Get all unique words
    words = set.union(*tokenized_texts)

    # Calculate IDF score for each word
    idf_scores = {word: np.log(len(df) / sum([word in text for text in tokenized_texts])) for word in tqdm(words, desc="Calculating IDF scores")}

    np.save("idf_scores.npy", idf_scores)


def make_w2v_embeddings(df, name, w2v_model):
    """
    Create word2vec embeddings for the given dataframe.

    Args:
        df (DataFrame): The dataframe which contains all the tweets.
        name (string): The name of the dataframe.
        w2v_model: Word2Vec model obtained from gensim.

    Returns:
        DataFrame: The dataframe with the embeddings.
    """

    features = []
    v_size = w2v_model.vector_size
    n_invalid_texts = 0
    n_only_unknown_words = 0

    # Calculate IDF scores if training data, otherwise load them
    if name == "train":
        calculate_idf_scores(df)

    idf_scores = np.load("idf_scores.npy", allow_pickle=True).item()

    for row in tqdm(df.itertuples(), total=len(df), desc=name):

        try:
            sentence = nltk.word_tokenize(row.text)
        except TypeError:
            # (e.g. empty sentence)
            n_invalid_texts += 1
            features.append(invalid_text_vec(v_size))
            continue

        # Determine the frequency of each word in the sentence
        tf_scores = {word: sentence.count(word) for word in sentence}

        # Get the word vectors using word2vec and weights using tf-idf
        word_vectors = []
        word_weights = []
        for word in sentence:
            try:

                # We use a temporary variable in case either the word embedding or the word weight is not found
                word_embedding = w2v_model[word]
                word_weight = idf_scores[word] * tf_scores[word]

                word_vectors.append(word_embedding)
                word_weights.append(word_weight)

            except KeyError:
                # ignore (e.g. unknown word)
                continue

        if len(word_vectors) == 0:
            n_only_unknown_words += 1
            features.append(invalid_text_vec(v_size))
            continue

        word_vectors = np.array(word_vectors)
        word_weights = np.array(word_weights)

        # Multiply each word vector with its weight and then take the mean
        weighted_vectors = word_vectors * word_weights[:, None]
        mean_vector = np.mean(weighted_vectors, axis=0)

        features.append(mean_vector)

    return np.array(features), n_invalid_texts, n_only_unknown_words


def get_tweet_embeddings_mobert(tweets, tokenizer, model):
    """
    Get the embeddings of the given tweets using the mobilebert model.
    Helper function for make_mobert_embeddings.

    Args:
        tweets (list): The tweets to get the embeddings of.
        tokenizer: The tokenizer of mobilebert
        model: The mobilebert model

    Returns:
        list: The embeddings of the tweets.
    """

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
    """
    Create mobilebert embeddings for the given dataframe.

    Args:
        df (DataFrame): The dataframe which contains all the tweets.
        name (string): The name of the dataframe.
        tokenizer: Mobilebert tokenizer.
        model: Mobilebert model.

    Returns:
        DataFrame: The dataframe with the embeddings.
    """

    tweets = df.text.values.tolist()

    embeddings = []
    BATCH_SIZE = 256
    for index in tqdm(range(0, len(tweets), BATCH_SIZE), desc=name):
        embeddings.append(get_tweet_embeddings_mobert(tweets[index:index+BATCH_SIZE], tokenizer, model))

    return torch.cat(embeddings).numpy()
