import pandas as pd
df=pd.read_csv("/Users/anandagarwal/reddit_mental_health_analysis/Clusters_Centers_text/2023_text.csv")
lda_df=pd.read_csv("/Users/anandagarwal/reddit_mental_health_analysis/Clusters_Centers_text/2023_document_topic_mapping.csv")
bert_df
filtered_df = df[df["text"] != "removed"]
final_df = filtered_df.drop_duplicates(subset=["text"])
print(len(final_df))
print(len(lda_df))