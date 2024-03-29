> This project is for the Machine Learning Practical ([WBAI060-05](https://ocasys.rug.nl/current/catalog/course/WBAI060-05)) course, as such we cannot accept outside contibutions at the moment.

# NLP Wizards: Emoji Prediction

In this project, we will be predicting the emoji for a given tweet. Our current approach uses a neural network classifier which is trained on the word2vec sentence embeddings of the tweets.

## Pre-requisites

To install all the dependencies, run the following command using pipenv:

```bash
pipenv install
```

## Project Structure

We currently have 3 main files in the project:

- `preprocess_dataset.py`: This file is used to clean the raw data and store it in a CSV file.

- `create_embedding.py`: This file is used to generate the sentence embeddings for the classifier. The embeddings are stored as numpy files.

- `create_model.py`: This file is used to train the classifier with a grid search and store the model.

## Running the project

To run the project, you can use the following commands:

```bash
pipenv shell
python preprocess_dataset.py
python create_embedding.py
python create_model.py
```

We will be consolidating all the commands into a single file in the future.

## Starting the API

To start the API, you can use the following command:

```bash
# It can take a while to start the API, so please be patient.
# It loads the model and the embeddings into memory.
uvicorn api:app --reload
```

You can access the Swagger Documentation at `http://127.0.0.1:8000/docs` and the API at `http://127.0.0.1:8000/get_emoji?text=...`.

## Starting front-end

To start the front-end, you can use the following command:
```bash
streamlit run streamlit_demo.py
```

## Authors

- Mansur Nurmukhambetov
- Jeremias Lino Ferrao
- Juriën Michèl Schut
