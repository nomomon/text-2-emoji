import streamlit as st
import requests

def handle_streamlit():
    title = "Emoji prediction"
    st.write(title)

    usr_sentence = st.text_input("Please enter a sentence:")

    api_url = "http://127.0.0.1:8000/get_emoji"
    usr_params = {'text': usr_sentence}
    try:
        data = requests.get(api_url, params=usr_params).json()
    except:
        st.write("API request failed")
        return

    # TODO: Put the output in a table
    for ix, result in enumerate(data["results"]):
        emoji = result["emoji"]
        prob = result["probability"]
        st.write(f"Number {ix}: \t{emoji}. \tProbability {prob}")


if __name__ == '__main__':
    handle_streamlit()
