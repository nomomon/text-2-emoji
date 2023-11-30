import re
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
