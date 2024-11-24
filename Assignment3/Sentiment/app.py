import streamlit as st
from textblob import TextBlob

# Set up the Streamlit app
st.title("Basic Sentiment Analysis")

# Input text
user_input = st.text_area("Enter text for sentiment analysis:")

# Perform sentiment analysis
if user_input:
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        st.success("Positive Sentiment 😊")
    elif sentiment < 0:
        st.error("Negative Sentiment 😞")
    else:
        st.info("Neutral Sentiment 😐")
