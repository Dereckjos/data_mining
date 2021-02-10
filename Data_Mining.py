from text_viz.viz import *
from model_viz.model import *
from model_viz.model_plot import *
import pandas as pd
import streamlit as st
from afinn import Afinn

df = pd.read_csv('small.csv', nrows=10000)
# df.head()
st.title('ABC News Headlines')
st.subheader('ABC News Dataset')
st.dataframe(df)
st.sidebar.title("Data Mining Tools :smile:")
text = st.sidebar.multiselect("Text Analytics",
                              ("Top 10 stop words", "Top 10 non stop words", "Word Cloud", "Ngram Plot"))

if text:
    if text[0] == "Top 10 stop words":
        st.subheader("Top 10 stop words")
        st.pyplot(plot_top_stopwords_barchart(df["headline_text"]))

    elif text[0] == "Top 10 non stop words":
        st.subheader("Top 10 non stop words")
        st.pyplot(plot_top_non_stopwords_barchart(df["headline_text"]))

    elif text[0] == "Word Cloud":
        st.subheader("Word Cloud")
        st.pyplot(plot_wordcloud(df["headline_text"]))

    else:
        st.header("Plotting N-Grams")
        st.subheader("Bigram Plot")
        st.pyplot(plot_top_ngrams_barchart(df["headline_text"], 2))
        st.subheader("Trigram Plot")
        st.pyplot(plot_top_ngrams_barchart(df["headline_text"], 3))
st.sidebar.header("Unsupervised Lexicon Models")
model = st.sidebar.multiselect("Select Model",
                               ("Afinn Lexicon Model", "TextBlob Model", "Vader Lexicon Model"))

if model:
    if model[0] == "Afinn Lexicon Model":
        st.subheader("Afinn Lexicon Model")
        af = afinnModel(df["headline_text"])
        st.write(af)
        st.subheader("Afinn Model Stats")
        st.pyplot(sentiment_plot(af["Predicted Sentiment"]))


    elif model[0] == "TextBlob Model":
        st.subheader("TextBlob Model")
        te = textblobModel(df["headline_text"])
        st.write(te)
        st.subheader("TextBlob Model Stats")
        st.pyplot(sentiment_plot(te["Predicted Sentiment"]))


    elif model[0] == "Vader Lexicon Model":
        st.subheader("Vader Lexicon Model")
        va = vaderModel(df["headline_text"])
        st.write(va)
        st.subheader("Vader Lexicon Stats")
        st.pyplot(sentiment_plot(va["Predicted Sentiment"]))

st.sidebar.header("All Model Comparison")

if st.sidebar.button("Click here to view model comparison"):
    st.header("Afinn Lexicon Model")
    af = afinnModel(df["headline_text"])
    st.subheader("Afinn Model Stats")
    st.pyplot(sentiment_plot(af["Predicted Sentiment"]))

    st.header("TextBlob Model Model")
    te = textblobModel(df["headline_text"])
    st.subheader("TextBlob Model Stats")
    st.pyplot(sentiment_plot(te["Predicted Sentiment"]))

    st.header("Vader Lexicon Model")
    va = vaderModel(df["headline_text"])
    st.subheader("Vader Lexicon Stats")
    st.pyplot(sentiment_plot(va["Predicted Sentiment"]))

st.sidebar.header("Sentiment Analysis")
choice = st.sidebar.multiselect("Select model for prediction",
                               ("Afinn Lexicon Model", "TextBlob Model"))
if choice:
    if choice[0] == "Afinn Lexicon Model":
        st.header("Sentiment Predictor")
        text = st.text_area("Enter Text Here")
        if st.button("Predict"):
            afn = Afinn()
            scores = afn.score(text)
            sentiment = ['Positive' if scores > 0
                         else 'Negative' if scores < 0
            else 'Neutral']

            if sentiment[0] == "Positive":
                st.header("Given Sentiment is {}".format(sentiment[0]))
                st.image('./positive.png')

            elif sentiment[0] == "Negative":
                st.header("Given Sentiment is {}".format(sentiment[0]))
                st.image('./negative.png')

            else:
                st.header("Given Sentiment is {}".format(sentiment[0]))
                st.image('./neutral.png')

    else:
        st.header("Sentiment Predictor")
        text = st.text_area("Enter Text Here")
        if st.button("Predict"):
            score = TextBlob(text).sentiment.polarity
            sentiment = ['Positive' if score > 0
                         else 'Negative' if score < 0
            else 'Neutral']

            if sentiment[0] == "Positive":
                st.header("Given Sentiment is {}".format(sentiment[0]))
                st.image('./positive.png')

            elif sentiment[0] == "Negative":
                st.header("Given Sentiment is {}".format(sentiment[0]))
                st.image('./negative.png')

            else:
                st.header("Given Sentiment is {}".format(sentiment[0]))
                st.image('./neutral.png')









    # if st.button("Predict"):
    #     afn = Afinn()
    #     scores = afn.score(text)
    #     sentiment = ['positive' if score > 0
    #                  else 'negative' if score < 0
    #     else 'neutral'
    #                  for score in scores]
    #
    #     if sentiment:
    #         st.success(sentiment)


# plot_top_non_stopwords_barchart(df["headline_text"])
# plot_top_ngrams_barchart(df['headline_text'],3)
# plot_top_ngrams_barchart(df['headline_text'],2)
# plot_wordcloud(df['headline_text'])
# af = afinnModel(df["headline_text"])
# te = textblobModel(df["headline_text"])
# va = vaderModel(df["headline_text"])
# afinn_graph = sentiment_plot(af["Predicted Sentiment"])
# textblob_graph = textblob_graph = sentiment_plot(te["Predicted Sentiment"])
# vader_graph = sentiment_plot(va["Predicted Sentiment"])
