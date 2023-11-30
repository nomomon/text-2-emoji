from project_name.data.text_processing import *
import pandas as pd
from tqdm import tqdm

import os

def read_raw(name, path):
    texts = []
    with open(path + name + "_text.txt", "r") as f:
        for line in f:
            texts.append(line)
    with open(path + name + "_labels.txt", "r") as f:
        labels = []
        for line in f:
            labels.append(line)
    return texts, labels

def make_df(texts, labels):
    df = pd.DataFrame({"text": texts, "label": labels})
    df["label"] = df["label"].apply(lambda x: x.strip())
    df["label"] = df["label"].apply(lambda x: int(x))
    return df

def make_df_from_raw(name="train", path="data/raw/"):
    texts, labels = read_raw(name, path)
    df = make_df(texts, labels)
    return df


if __name__ == '__main__':
    print('Installing nltk...')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    print('Making directories...')
    os.makedirs('../data/bronze', exist_ok=True)
    os.makedirs('../data/silver', exist_ok=True)
    os.makedirs('../data/gold', exist_ok=True)

    print('Reading raw data...')
    make_df_from_raw("train").to_csv("../data/bronze/train.csv", index=False)
    make_df_from_raw("test").to_csv("../data/bronze/test.csv", index=False)
    make_df_from_raw("val").to_csv("../data/bronze/valid.csv", index=False)

    print('Preprocessing data...')
    train = pd.read_csv('./data/bronze/train.csv')
    valid = pd.read_csv('./data/bronze/valid.csv')
    test = pd.read_csv('./data/bronze/test.csv')
    
    for df, name in [(train, "train"), (valid, "valid"), (test, "test")]:
        for row in tqdm(df.itertuples(), total=len(df), desc=name):
            df.at[row.Index, 'text'] = preprocess_text(row.text)

    train.to_csv('./data/silver/train.csv', index=False)
    valid.to_csv('./data/silver/valid.csv', index=False)
    test.to_csv('./data/silver/test.csv', index=False)

    # TODO: Add code to create gold dataset here
    #       (e.g. create a dataset with embeddings)

    print('Finished preprocessing data.')