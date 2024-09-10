import pandas as pd
import os

def file_process(data_df, submissions_df, cluster_df):
    # Initialize the final DataFrame with necessary columns
    final_df = pd.DataFrame(columns=["author", "title", "text", "lda_topic", "cluster"])
    
    # Copy data from submissions_df and data_df to final_df
    final_df["author"] = submissions_df["author"].copy()
    final_df["title"] = submissions_df["title"].copy()
    final_df["text"] = submissions_df["text"].copy()
    final_df["lda_topic"] = data_df["Dominant Topic"].copy()

    # Merge final_df with cluster_df on 'author' to include only the 'cluster' column
    final_df = final_df.merge(cluster_df[['author', 'cluster']], on='author', how='left')

    return final_df

def main():
    years = [2020]
    for year in years:
        print(f"Processing year: {year}")
        data_df = pd.read_csv(f"/Users/anandagarwal/reddit_mental_health_analysis/Clusters_Centers_text/{year}_document_topic_mapping.csv")
        submissions_df = pd.read_csv(f"/Users/anandagarwal/reddit_mental_health_analysis/year_wise_filtered_submissions/filtered_submissions_{year}.csv")
        cluster_df = pd.read_csv(f"/Users/anandagarwal/reddit_mental_health_analysis/users_with_number_of_posts/year_{year}.csv")
        
        print("Dataframes loaded successfully.")
        final_df = file_process(data_df, submissions_df, cluster_df)
        
        output_path = f'/Users/anandagarwal/reddit_mental_health_analysis/lda_final_data/year_{year}.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"File saved: {output_path}")

if __name__ == "__main__":
    main()
