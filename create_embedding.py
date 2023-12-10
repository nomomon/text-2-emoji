import nltk
import gensim.downloader as w2v_api
from tqdm import tqdm
import pandas as pd
import numpy as np
# from sklearn.preprocessing import normalize


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


def make_sentence_embeddings(encoder_type="word2vec"):
    train = pd.read_csv('./data/silver/train.csv')
    valid = pd.read_csv('./data/silver/valid.csv')
    test = pd.read_csv('./data/silver/test.csv')

    if encoder_type == "word2vec":
        model = w2v_api.load("word2vec-google-news-300")
        v_size = model.vector_size

    n_invalid_texts = 0
    n_only_unknown_words = 0
    for df, name in [(train, "train"), (valid, "valid"), (test, "test")]:

        if encoder_type == "word2vec":
            features, set_invalid_texts, set_unknown_words = make_w2v_embeddings(df, name, model)
            target = df.label.values

            n_invalid_texts += set_invalid_texts
            n_only_unknown_words += set_unknown_words

        # Normalize the features to unit length
        # Seems like this results in worse performance
        # features = normalize(features)

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
    make_sentence_embeddings()
    print("done")
