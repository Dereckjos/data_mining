import matplotlib.pyplot as plt
import pandas as pd


def sentiment_plot(text):
    senti = pd.Series(text)
    plt.bar(senti.value_counts().index,
            senti.value_counts())

