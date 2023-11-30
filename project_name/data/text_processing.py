import re

import nltk
from nltk.corpus import stopwords

def clean_text(text):
    """
        Clean text by removing unnecessary characters (non-aphabetic) and multiplying spaces

        Args:
            text (str): Text to clean
        Returns:
            str: Cleaned text
    """

    # Remove non-alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text)

    # Remove multiple spaces
    text = re.sub('\s+', ' ', text)

    return text

def remove_stopwords(text):
    """
        Remove stopwords from text

        Args:
            text (str): Text to remove stopwords from
        Returns:
            str: Text without stopwords
    """

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Reconstruct text
    text = ' '.join(tokens)

    return text

def lemmatize_text(text):
    """
        Lemmatize text

        Args:
            text (str): Text to lemmatize
        Returns:
            str: Lemmatized text
    """

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Lemmatize text
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Reconstruct text
    text = ' '.join(tokens)

    return text

def preprocess_text(text):
    """
        Preprocess text by cleaning, removing stopwords and lemmatizing

        Args:
            text (str): Text to preprocess
        Returns:
            str: Preprocessed text
    """

    # Clean text
    text = clean_text(text)

    # Remove stopwords
    text = remove_stopwords(text)

    # Lemmatize text
    text = lemmatize_text(text)

    # Lowercase text
    text = text.lower()

    return text