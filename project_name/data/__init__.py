from text_processing import *
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    print('Preprocessing data...')

    train = pd.read_csv('./data/bronze/train.csv')
    valid = pd.read_csv('./data/bronze/valid.csv')
    test = pd.read_csv('./data/bronze/test.csv')
    
    # Preprocess text
    for df in [train, valid, test]:
        for row in tqdm(df.itertuples(), total=len(df)):
            df.at[row.Index, 'text'] = preprocess_text(row.text)


    train.to_csv('./data/silver/train.csv', index=False)
    valid.to_csv('./data/silver/valid.csv', index=False)
    test.to_csv('./data/silver/test.csv', index=False)