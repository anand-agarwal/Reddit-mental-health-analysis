import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_topic_distributions(df, year):
    # Debug: Check if 'cluster' column exists
    # if 'cluster' not in df.columns:
    #     print(f"Error: 'cluster' column not found in DataFrame for year {year}")
    #     print(df.head())  # Print the first few rows of the DataFrame for inspection
    #     return

    clusters = [0,1,2,3]
    for cluster in clusters:
        cluster_df = df[df['cluster_y'] == cluster]
        
        # Debug: Check if 'lda_topic' column exists
        if 'lda_topic' not in cluster_df.columns:
            print(f"Error: 'lda_topic' column not found in DataFrame for cluster {cluster} in year {year}")
            print(cluster_df.head())  # Print the first few rows of the DataFrame for inspection
            continue

        # Group by topic to get the count of documents in each topic for this cluster
        topic_distribution = cluster_df['lda_topic'].value_counts().sort_index()

        # Debug: Check if topic_distribution is empty
        if topic_distribution.empty:
            print(f"Warning: No topic distribution data for cluster {cluster} in year {year}")
            continue
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Use a try-except block to catch any issues during plotting
        try:
            topic_distribution.plot(kind='bar', color='skyblue')
            plt.title(f'Topic Distribution for Cluster {cluster} in Year {year}')
            plt.xlabel('Topic')
            plt.ylabel('Number of Documents')
            plt.xticks(rotation=0)
            plt.grid(axis='y')
            
            # Create the output directory if it doesn't exist
            output_dir = f'/Users/anandagarwal/reddit_mental_health_analysis/lda_plots/{year}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the plot as a PNG file
            output_path = os.path.join(output_dir, f'cluster_{cluster}_topic_distribution.png')
            plt.savefig(output_path)
            plt.close()
            print(f'Saved plot to {output_path}')
        except Exception as e:
            print(f"Error plotting data for cluster {cluster} in year {year}: {e}")

def main():
    years = [2020, 2021, 2022, 2023]
    for year in years:
        print(f"Processing year: {year}")
        final_df = pd.read_csv(f'/Users/anandagarwal/reddit_mental_health_analysis/lda_final_data/year_{year}.csv')
        
        # Debug: Check the structure and content of the DataFrame
        print(f"Dataframe for year {year} loaded successfully with columns: {final_df.columns.tolist()}")
        print(final_df.head())
        
        plot_topic_distributions(final_df, year)

if __name__ == "__main__":
    main()
