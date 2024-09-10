import pandas as pd
import os
import re 
from wordcloud import STOPWORDS

def remove_emojis(text):
    emoji_pattern=re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U00002702-\U000027B0"  # Dingbats
                           u"\U000024C2-\U0001F251" 
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r"", text)


def remove_stopwords(text):
    stop_words = set(STOPWORDS)
    text = str(text)
    processed_text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return processed_text

def file_process( bert_df, text_df, cluster_df): 
    final_df = pd.DataFrame(columns=["author", "text", "lda_topic", "bert_topic", "cluster"])
    final_df["author"] = text_df["author"].copy()
    final_df["text"] = text_df["combined_text"].copy()
    final_df = final_df.merge(cluster_df[['author', 'cluster']], on='author', how='left')
    final_df=final_df.drop(columns=["cluster_x"])
    final_df["bert_topic"] = bert_df["Topic Words"].copy()
    
    
    
    return final_df

def stats(data_df): 
    df = pd.DataFrame(columns=["lda_topic", "bert_topic", "count"])
    for i in range(10): 
        int_df = data_df[data_df["lda_topic"] == float(i)]
        print(f"Processing LDA topic {i}, number of entries: {len(int_df)}")  # Debug statement
        if not int_df.empty:
            bert_topic_counts = int_df["bert_topic"].value_counts().reset_index()
            bert_topic_counts.columns = ["bert_topic", "count"]
            bert_topic_counts["lda_topic"] = i
            print(f"BERT topic counts for LDA topic {i}:\n{bert_topic_counts}")  # Debug statement
            df = pd.concat([df, bert_topic_counts], ignore_index=True)
            print(f"Concatenated DataFrame after LDA topic {i}:\n{df}")  # Debug statement
    
    return df

def main(): 
    years = [2020, 2021, 2022, 2023]
    for year in years: 
        print(f"Processing year: {year}")
        lda_df = pd.read_csv(f"Clusters_Centers_text/{year}_document_topic_mapping.csv")
        bert_df = pd.read_csv(f"bert_topic_with_docid_{year}.csv")
        cluster_df=pd.read_csv(f"users_with_number_of_posts/year_{year}.csv")
        df=pd.read_csv(f"/Users/anandagarwal/reddit_mental_health_analysis/year_wise_filtered_submissions/filtered_submissions_{year}.csv")
        
        df["title"]=df["title"].fillna("").astype(str)
        df=df[df["text"]!="removed"]
        df['text'] = df['text'].astype(str)
        df['title'] = df['title'].astype(str)
        
        df["combined_text"]=df[["title", "text"]].agg("-".join, axis=1)
        df['combined_text'] = df['combined_text'].str.replace('\n\s*\n', '', regex=True)


        df["combined_text"]=df["combined_text"].apply(remove_emojis)
        filtered_df = df[df["combined_text"] != "removed"]
        fina_df = filtered_df.drop_duplicates(subset=["combined_text"])
        fina_df['combined_text'] = fina_df['combined_text'].astype(str)
        fina_df["text"] = fina_df["text"].apply(remove_stopwords)
        
        
        
        print(len(lda_df))
        print(len(bert_df))
        print(len(fina_df))
        
        #Debug statements
        print(f"LDA DataFrame columns: {lda_df.columns}")
        print(f"BERT DataFrame columns: {bert_df.columns}")
        
        final_df = file_process(bert_df, fina_df, cluster_df)
        final_df["lda_topic"] = lda_df["Dominant Topic"].copy()
        stats_df = stats(final_df)
        
        output_path = f"topics_stats/year_{year}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        stats_output_path = f"topics_stats_freq/year_{year}.csv"
        os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
        stats_df.to_csv(stats_output_path, index=False)
        
        # Debug statement
        print(f"Saved stats for year {year}, number of entries: {len(stats_df)}")

if __name__ == "__main__":
    main()

