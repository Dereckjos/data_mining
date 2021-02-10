from afinn import Afinn
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


def afinnModel(text):
    afn = Afinn()
    scores = [afn.score(article) for article in text]
    sentiment = ['positive' if score > 0
                 else 'negative' if score < 0
    else 'neutral'
                 for score in scores]

    dict = {"News Headlines": text, "Predicted Sentiment": sentiment}
    afinn_pred = pd.DataFrame(dict)

    return afinn_pred


def textblobModel(text):
    x = [TextBlob(te).sentiment.polarity for te in text]
    sentiment = ['positive' if score > 0
                 else 'negative' if score < 0
    else 'neutral'
                 for score in x]

    dict = {"News Headlines": text, "Predicted Sentiment": sentiment}
    textblob_pred = pd.DataFrame(dict)

    return textblob_pred


def vaderModel(text):
    si = SentimentIntensityAnalyzer()
    scores = [si.polarity_scores(article) for article in text]
    sc = [{k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)} for score in scores]
    sentiment = [list(i.keys())[0] for i in sc]
    dict1 = {"News Headlines": text, "Predicted Sentiment": sentiment}
    vader_pred = pd.DataFrame(dict1)
    return vader_pred

# df = pd.DataFrame([{'text':'I love the movie'},{"text":'I hate the movie'}])
#
# af = afinnModel(df["text"])
# tx = textblobModel(df["text"])
# va = vaderModel(df["text"])
