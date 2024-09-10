import pandas as pd
import spacy
from gensim import corpora, models
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from tqdm import tqdm
from multiprocessing import cpu_count
import logging

# Download necessary resources
# nltk.download('wordnet')
# nltk.download('stopwords')
# spacy.cli.download("en_core_web_md")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Combine NLTK and wordcloud stopwords
stop_words = set(stopwords.words('english')).union(STOPWORDS)

# Remove stopwords function
def remove_stopwords(text):
    text = str(text)
    return " ".join(word for word in text.split() if word.lower() not in stop_words)

# Load spaCy model
nlp = spacy.load("en_core_web_md", disable=['parser', 'ner'])

# Lemmatization function with error handling
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ']):
    output = []
    for doc in tqdm(nlp.pipe(texts, batch_size=50, n_process=cpu_count() - 1), total=len(texts)):
        try:
            output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            output.append([])
    return output

# Load data
df = pd.read_csv("/Users/anandagarwal/reddit_mental_health_analysis/year_wise_filtered_submissions/filtered_submissions_2020.csv")

filtered_df = df[df["text"] != "removed"]
final_df = filtered_df.drop_duplicates(subset=["text"])

# Remove stopwords
final_df["text"] = final_df["text"].apply(remove_stopwords)

# Split data into chunks of 1000 rows each
chunk_size = 1000
chunks = [final_df.iloc[i:i + chunk_size] for i in range(0, len(final_df), chunk_size)]

# Function to train LDA model on a chunk of data
def train_lda_on_chunk(chunk):
    text_list = chunk['text'].tolist()
    tokenized_reviews = lemmatization(text_list)
    dictionary = corpora.Dictionary(tokenized_reviews)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]
    lda_model = models.LdaMulticore(corpus=doc_term_matrix, id2word=dictionary, random_state=100,
                                    chunksize=1000, passes=10, iterations=50, workers=cpu_count() - 1)
    return lda_model, tokenized_reviews

# Train LDA models on each chunk and collect intermediate results
all_tokenized_reviews = []
all_models = []

for chunk in chunks:
    lda_model, tokenized_reviews = train_lda_on_chunk(chunk)
    all_tokenized_reviews.extend(tokenized_reviews)
    all_models.append(lda_model)

# Create final dictionary and document-term matrix from all tokenized reviews
final_dictionary = corpora.Dictionary(all_tokenized_reviews)
final_doc_term_matrix = [final_dictionary.doc2bow(rev) for rev in all_tokenized_reviews]

# Build final LDA model using combined data
final_lda_model = models.LdaMulticore(corpus=final_doc_term_matrix, id2word=final_dictionary, random_state=100,
                                      chunksize=1000, passes=50, iterations=100, workers=cpu_count() - 1)

# Print LDA topics
topics = final_lda_model.print_topics()
print(topics)

# Calculate perplexity
perplexity = final_lda_model.log_perplexity(final_doc_term_matrix)

# Calculate coherence
coherence_model_lda = models.CoherenceModel(model=final_lda_model, texts=all_tokenized_reviews, dictionary=final_dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

# Save results to a text file
with open("/Users/anandagarwal/reddit_mental_health_analysis/LDA_results/2021.text", "w") as f:
    f.write("LDA Topics:\n")
    f.write("\n".join([f"Topic {i}: {topic}" for i, topic in topics]) + "\n\n")
    f.write(f"Perplexity: {perplexity}\n")
    f.write(f"Coherence: {coherence_lda}\n")

logger.info("Results saved successfully.")


