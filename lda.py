import numpy as np
import pandas as pd
import random
import logging
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from tqdm import tqdm
from multiprocessing import cpu_count, freeze_support
import spacy
from gensim import corpora, models

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Combine NLTK and wordcloud stopwords
stop_words = set(STOPWORDS)

# Remove stopwords function
def remove_stopwords(text):
    text = str(text)
    return " ".join(word for word in text.split() if word.lower() not in stop_words)

# Load spaCy model
nlp = spacy.load("en_core_web_md", disable=['parser', 'ner'])

# Lemmatization function with error handling
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ']):
    output = []
    for doc in tqdm(nlp.pipe(texts, batch_size=50, n_process=1), total=len(texts)):
        try:
            output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            output.append([])
    return output

def process_year(year):
    # Load data
    file_path = f"/Users/anandagarwal/reddit_mental_health_analysis/Clusters_Centers_text/{year}_text.csv"
    df = pd.read_csv(file_path)

    filtered_df = df[df["text"] != "removed"]
    final_df = filtered_df.drop_duplicates(subset=["text"])

    # Remove stopwords
    final_df["text"] = final_df["text"].apply(remove_stopwords)

    # Get list of texts
    text_list = final_df["text"].tolist()

    # Lemmatize texts
    tokenized_reviews = lemmatization(text_list)
    dictionary = corpora.Dictionary(tokenized_reviews)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]

    # Build LDA model using all data with a reduced number of topics and chunksize
    num_topics = 10  # Start with a smaller number of topics
    lda_model = models.LdaMulticore(corpus=doc_term_matrix, id2word=dictionary, num_topics=10,
                                    random_state=100, chunksize=1000, passes=10, iterations=70, workers=1)

    # Print and generate word cloud for LDA topics
    topics = lda_model.print_topics()
    print(f"Topics for year {year}: {topics}")

    # Get the representative documents for each topic
    topic_docs = {i: [] for i in range(num_topics)}
    for doc_id, bow in enumerate(doc_term_matrix):
        topic_dist = lda_model.get_document_topics(bow)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
        topic_docs[dominant_topic].append(final_df.iloc[doc_id]["text"])

    # Save the results to a CSV file
    results = []
    for topic_id, topic_terms in topics:
        representative_docs = topic_docs[topic_id][:5]  # Take top 5 representative documents
        results.append([topic_id, topic_terms, representative_docs])

    results_df = pd.DataFrame(results, columns=["Topic ID", "Terms", "Representative Documents"])
    output_file = f"/Users/anandagarwal/reddit_mental_health_analysis/Clusters_Centers_text/{year}_lda_results.csv"
    results_df.to_csv(output_file, index=False)

def main():
    years = [2022, 2023]
    for year in years:
        process_year(year)

if __name__ == "__main__":
    main()


