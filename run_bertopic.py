import pandas as pd
from bertopic import BERTopic
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
from sentence_transformers import SentenceTransformer
from  umap.umap_ import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.colors as mcolors
import plotly.express as px

def remove_stopwords(text):
    stop_words = set(STOPWORDS)
    text = str(text)
    processed_text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return processed_text

def build_bert_model(data, year):
    data = data.drop_duplicates(subset=["text"])
    data["text"]=data["text"].str.replace("removed", "")
    data = data[data["text"] != "removed"]
    data['text'] = data['text'].astype(str)
    data["text"] = data["text"].apply(remove_stopwords)

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model=UMAP(n_neighbors=5, n_components=5, min_dist=0.5)
    hdbscan_model = KMeans(n_clusters=50)
    vectorizer_model = CountVectorizer(ngram_range=(1, 2))
    representation_model=KeyBERTInspired()
    
    
    
    embeddings=sentence_model.encode(data["text"].tolist(), show_progress_bar=True)
    
    topic_model = BERTopic(embedding_model=sentence_model,
                           umap_model=umap_model,  
                           vectorizer_model=vectorizer_model,
                           hdbscan_model=hdbscan_model, 
                           representation_model=representation_model,
                           top_n_words=10,
                           language='english',
                           calculate_probabilities=True,
                           verbose=True)
    
    
    topics, probs = topic_model.fit_transform(data["text"].tolist(), embeddings=embeddings)
    info = topic_model.get_topic_info()
    print(info)
    
    tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(embeddings) 
    unique_topics = np.unique(topics)
    num_topics = len(unique_topics)
    cmap = plt.get_cmap('tab20', num_topics)
    colors = [cmap(i) for i in range(num_topics)]
    
    # Assign a color to each topic
    topic_colors = [colors[topic] for topic in topics]

    # Plotting
    df_tsne = pd.DataFrame(X_tsne, columns=['x', 'y', 'z'])
    df_tsne['topic'] = topics

    # Create an interactive plot using Plotly
    fig = px.scatter_3d(df_tsne, x='x', y='y', z='z', color=df_tsne['topic'].astype(str),
                     title=f"t-SNE Plot of Embeddings for Year {year}",
                     labels={'color': 'Topic'},
                     hover_data=[data['text']])
    
   
    fig.write_html(f"tsne_plot_{year}.html")
    
    
    info.to_csv(f"Bert_topics/{year}.csv", index=False)
    return topics, topic_model, probs

def doc_with_ids(data, topics, topic_model, year):
    ids = data.index.tolist()
    results = {id: topic_model.get_topic(topic) for id, topic in zip(ids, topics)}
    results_list = []
    
    for doc_id, topic in results.items():
        topic_words = ", ".join([word for word, _ in topic])
        results_list.append({"Document ID": doc_id, "Topic Words": topic_words})
        
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f"bert_topic_with_docid_{year}.csv", index=False)

def topic_vis(topic_model, year):
    heatmap = topic_model.visualize_heatmap()
    heatmap.write_html(f"bert_topic_heatmaps/{year}.html")

years = [ 2020, 2021, 2022, 2023]
for year in years:
    df = pd.read_csv(f"Clusters_Centers_text/{year}_text.csv")
    topics=[]

    topics, topic_model, probs = build_bert_model(df, year)
    #doc_with_ids(df, topics, topic_model, year)
    #topic_vis(topic_model, year)
