import pandas as pd

# File paths
file_path = "users_with_number_of_posts/year_2020_final.csv"
submission_file_path = "year_wise_filtered_submissions/filtered_submissions_2020.csv"
comments_file_path = "/Users/anandagarwal/reddit_mental_health_analysis/year_wise_filtered_comments/filtered_comments_2020.csv"

# Read the users DataFrame
df = pd.read_csv(file_path)

# Read the submissions and comments data
submissions_df = pd.read_csv(submission_file_path)
comments_df = pd.read_csv(comments_file_path)

# Ensure 'text' and 'score' columns exist and all 'text' values are strings
assert 'text' in submissions_df.columns and 'score' in submissions_df.columns, "Required columns are not present in the submissions DataFrame"
assert 'body' in comments_df.columns and 'score' in comments_df.columns, "Required columns are not present in the comments DataFrame"
submissions_df["text"] = submissions_df["text"].fillna("").astype(str)
comments_df["body"] = comments_df["body"].fillna("").astype(str)

# Group by author and concatenate all text for each author
submission_texts = submissions_df.groupby("author")["text"].apply(lambda texts: " ".join(texts)).reset_index()
comment_texts = comments_df.groupby("author")["body"].apply(lambda texts: " ".join(texts)).reset_index()

# Function to count unique words in a text
def count_unique_words(text):
    words = set(text.split())  # Split the text into words and store them in a set
    return len(words)          # Return the number of unique words

# Calculate the number of unique words for each author in posts and comments
submission_texts["number_of_unique_words_in_posts"] = submission_texts["text"].apply(count_unique_words)
comment_texts["number_of_unique_words_in_comments"] = comment_texts["body"].apply(count_unique_words)

# Calculate the highest score for each author in posts and comments
submission_scores = submissions_df.groupby("author")["score"].max().reset_index()
comment_scores = comments_df.groupby("author")["score"].max().reset_index()

# Rename columns
submission_scores = submission_scores.rename(columns={"score": "highest_score_in_posts"})
comment_scores = comment_scores.rename(columns={"score": "highest_score_in_comments"})

# Merge the unique word counts and scores with the original users DataFrame
df = df.merge(submission_texts[["author", "number_of_unique_words_in_posts"]], on="author", how="left")
df = df.merge(comment_texts[["author", "number_of_unique_words_in_comments"]], on="author", how="left")
df = df.merge(submission_scores[["author", "highest_score_in_posts"]], on="author", how="left")
df = df.merge(comment_scores[["author", "highest_score_in_comments"]], on="author", how="left")

# Replace NaNs with 0
df["number_of_unique_words_in_posts"] = df["number_of_unique_words_in_posts"].fillna(0).astype(int)
df["number_of_unique_words_in_comments"] = df["number_of_unique_words_in_comments"].fillna(0).astype(int)
df["highest_score_in_posts"] = df["highest_score_in_posts"].fillna(0).astype(int)
df["highest_score_in_comments"] = df["highest_score_in_comments"].fillna(0).astype(int)

# Save the updated DataFrame to a CSV file
df.to_csv(file_path, index=False)

# Print the updated DataFrame
print(df)



