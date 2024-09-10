import os
import logging
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim import corpora, models

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def save_topics_to_file_and_wordcloud(year):
    # Load dictionary and LDA model
    dictionary_path = f'/Users/anandagarwal/reddit_mental_health_analysis/lda_dictionaries/{year}_dictionary.gensim'
    model_path = f'/Users/anandagarwal/reddit_mental_health_analysis/lda_models/{year}_lda_model.gensim'

    dictionary = corpora.Dictionary.load(dictionary_path)
    lda_model = models.LdaModel.load(model_path)

    # Print topics and word distributions
    topics = lda_model.print_topics(num_words=10)
    topics_output = []
    for topic_id, topic in topics:
        topic_string = f'Topic {topic_id}: {topic}'
        print(topic_string)
        topics_output.append(topic_string)

    # Save topics to a text file
    output_file_path = f'/Users/anandagarwal/reddit_mental_health_analysis/lda_topics/{year}_topics.txt'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as f:
        for line in topics_output:
            f.write(f"{line}\n")

    # Generate word clouds for each topic
    wordcloud_output_dir = f'/Users/anandagarwal/reddit_mental_health_analysis/lda_wordclouds/{year}/'
    os.makedirs(wordcloud_output_dir, exist_ok=True)
    for topic_id, topic in topics:
        word_freq = {word.strip(' "'): float(weight) for weight, word in [pair.split('*') for pair in topic.split(' + ')]}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_id}')
        plt.savefig(f'{wordcloud_output_dir}/topic_{topic_id}.png')
        plt.close()

def main():
    years = [2020, 2021, 2022, 2023]
    for year in years:
        save_topics_to_file_and_wordcloud(year)

if __name__ == "__main__":
    main()
