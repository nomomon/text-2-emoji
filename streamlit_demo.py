import streamlit as st
import requests
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def handle_streamlit():
    """
    Creates a streamlit frontend. Receives a json file from the API
    containing emojis and probabilities, and presents them in a table.

    Returns:
         None
    """
    title = "Text-2-Emoji"
    st.write(title)

    usr_sentence = st.text_input("Please enter a sentence:")

    api_url = "http://127.0.0.1:8000/get_emoji"
    usr_params = {'text': usr_sentence}
    try:
        data = requests.get(api_url, params=usr_params).json()
    except requests.exceptions.RequestException as e:
        st.write(f"API request failed. Exception type: {type(e).__name__}")
        return

    df_data = pd.json_normalize(data["results"])

    custom_darkcyan = ["#18191A", "darkcyan"]    # Probability background color based on its value
    custom_cmap = LinearSegmentedColormap.from_list("mycmap",custom_darkcyan)
    df_data = df_data.style.background_gradient(axis=0, subset="probability",
                                                vmin=0, vmax=1, cmap=custom_cmap)
    st.table(df_data)


if __name__ == '__main__':
    handle_streamlit()
