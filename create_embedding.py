import nltk
from gensim.models import Word2Vec
from tqdm import tqdm
import pandas as pd
import numpy as np


def invalid_sentence_vec(v_size):
    return np.zeros(v_size, dtype=np.float32)


def make_sentence_embeddings():
    train = pd.read_csv('./data/silver/train.csv')
    valid = pd.read_csv('./data/silver/valid.csv')
    test = pd.read_csv('./data/silver/test.csv')

    corpus = []
    for df, name in [(train, "train"), (valid, "valid")]:
        for row in tqdm(df.itertuples(), total=len(df), desc=name):
            corpus.append(nltk.word_tokenize(row.text))

    v_size = 100  # 100 is default
    w2v_model = Word2Vec(corpus, min_count=1, vector_size=v_size)

    print("Model created, creating embeddings...")

    empty_rows = 0
    only_unknown_words = 0
    for df, name in [(train, "train"), (valid, "valid"), (test, "test")]:
        for row in tqdm(df.itertuples(), total=len(df), desc=name):
            try:
                sentence = nltk.word_tokenize(row.text)
            except TypeError:  # Empty sentence for example
                empty_rows += 1
                df.at[row.Index, 'text'] = invalid_sentence_vec(v_size)
                continue

            word_vectors = []
            for word in sentence:
                try:
                    word_vectors.append(w2v_model.wv[word])
                except KeyError:  # Word not trained on, ignore
                    continue

            if len(word_vectors) == 0:
                only_unknown_words += 1
                df.at[row.Index, 'text'] = invalid_sentence_vec(v_size)
                continue

            word_vectors = np.array(word_vectors)
            mean_vector = word_vectors.mean(axis=0)
            df.at[row.Index, 'text'] = mean_vector

    print(f"Empty sentences encountered: {empty_rows}\n"
          f"Sentences with only unknown words: {only_unknown_words}")

    print("to csv.....")
    train.to_csv('./data/gold/train.csv', index=False)
    valid.to_csv('./data/gold/valid.csv', index=False)
    test.to_csv('./data/gold/test.csv', index=False)

if __name__ == '__main__':
    make_sentence_embeddings()
    print("done")
