import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to perform clustering and plot results
def cluster_and_plot(year, k=4):
    file_path = f"/Users/anandagarwal/reddit_mental_health_analysis/users_with_number_of_posts/year_{year}.csv"
    data = pd.read_csv(file_path)
    data.fillna(0, inplace=True)

    # Select features for clustering
    features = data.drop(columns=['author', 'number_of_unique_words_in_posts', 'number_of_unique_words_in_comments', 
                                  "highest_score_in_posts", "highest_score_in_comments", "cluster"])

    # Scale the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=43)
    data['cluster'] = kmeans.fit_predict(scaled_features)

    # Get cluster centers
    cluster_centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
    number_of_posts = cluster_centers_df["number_of_posts"]
    number_of_comments = cluster_centers_df["number_of_comments"]
    data.to_csv(file_path, index=False)

    # Plotting the results
    clusters = range(k)
    width = 0.1
    x = np.arange(len(clusters))

    fig, ax = plt.subplots()
    ax.bar(x - width * 2.5, number_of_posts, width, label="Number of Posts")
    ax.bar(x - width * 1.5, number_of_comments, width, label="Number of Comments")

    # Adding labels and title
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Values")
    ax.set_title(f"Year {year}")
    ax.set_xticks(x)
    ax.set_xticklabels(clusters)
    ax.legend()

    # Create directory if it does not exist
    output_dir = f"cluster_graphs"
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(output_dir, f"cluster_plot_{year}.png"))
    plt.close()

# Loop through the years and process each file
for year in [2020, 2021, 2022, 2023]:
    cluster_and_plot(year)
