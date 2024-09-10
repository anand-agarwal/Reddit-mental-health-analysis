import csv
import pandas as pd

# Input file path
input_file = "/Users/anandagarwal/reddit_mental_health_analysis/Clusters_Centers_text/2023/cluster_0.csv"

# Load the CSV file
with open(input_file, "w", newline='') as fin:
    writer = csv.writer(fin)
    writer.writerow(["author", "text"])
    
    # Load the text data
    text_df = pd.read_csv("year_wise_filtered_submissions/filtered_submissions_2020.csv")
    authors_df = pd.read_csv("/Users/anandagarwal/reddit_mental_health_analysis/users_with_number_of_posts/year_2020.csv")

    # Group by 'author' and convert text and float instances to strings before joining
    new_df = text_df.groupby("author")['text']
    texts = new_df.apply(lambda texts: " ".join(map(str, texts)))

    # Find the column index for 'cluster' and 'author'
    cluster_index = authors_df.columns.get_loc("cluster")
    author_index = authors_df.columns.get_loc("author")

    # Filter authors belonging to cluster 3
    authors = authors_df[authors_df.iloc[:, cluster_index] == 0].iloc[:, author_index].tolist()

    # Write rows where authors are in the cluster 3 authors list
    for author, text in texts.items():
        if author in authors:
            writer.writerow([author, text])
            
print(writer)

