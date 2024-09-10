import pandas as pd
import os 
from sentence_transformers import SentenceTransformer



def remove_stopwords(text):
    stop_words = set(STOPWORDS)
    text = str(text)
    processed_text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return processed_text


def build_tsne_plot(data, year): 
    data = data.drop_duplicates(subset=["text"])
    data["text"]=data["text"].str.replace("removed", "")
    data = data[data["text"] != "removed"]
    data['text'] = data['text'].astype(str)
    data["text"] = data["text"].apply(remove_stopwords)

years = [2020]
for year in years:
    df = pd.read_csv(f"Clusters_Centers_text/{year}_text.csv")
    topics=[]

    topics, topic_model, probs = build_tsne_plot(df, year)