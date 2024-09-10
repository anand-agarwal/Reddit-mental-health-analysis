import re
import pandas as pd 



df=pd.read_csv("/Users/anandagarwal/reddit_mental_health_analysis/year_wise_filtered_submissions/filtered_submissions_2020.csv")
df["title"]=df["title"].fillna("").astype(str)
df=df[df["text"]!="removed"]
df['text'] = df['text'].astype(str)
df['title'] = df['title'].astype(str)
print(df)

new_df=pd.DataFrame(columns=["text"])
new_df["text"]=df[["title", "text"]].agg("-".join, axis=1)
new_df['text'] = new_df['text'].str.replace('\n\s*\n', '', regex=True)
print(new_df)

filtered_df = new_df[new_df["text"] != "removed"]
fina_df = filtered_df.drop_duplicates(subset=["text"])


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

new_df["text"]=new_df["text"].apply(remove_emojis)
print(len(new_df))
# new_df.to_csv("Clusters_Centers_text/2023_text.csv")

