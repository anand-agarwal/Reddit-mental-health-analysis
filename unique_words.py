"""import pandas as pd
import csv

file_path = "users_with_number_of_posts/year_2020.csv"
submission_file_path = "year_wise_filtered_submissions/filtered_submissions_2020.csv"
comments_file_path="/Users/anandagarwal/reddit_mental_health_analysis/year_wise_filtered_comments/filtered_comments_2020.csv"

# Read the users DataFrame
df = pd.read_csv(file_path)

def text_compiler(file_path, author):
    text = ""
    with open(file_path, "r") as fin:
        rows = csv.reader(fin)
        header = next(rows)
        text_index = header.index("text")
        author_index = header.index("author")
        for row in rows:
            if row[author_index] == author:
                text += " " + row[text_index]
    return text

def unique_words(text):
    words = set()
    for word in text.split():
        words.add(word)
    return len(words)

# Initialize the new column in the DataFrame
df["number_of_unique_words"] = 0

# Update the DataFrame with the number of unique words for each author
for i in range(len(df)):     
    author = df.at[i, 'author']
    text = text_compiler(submission_file_path, author)
    number_unique_words = unique_words(text)
    df.at[i, 'number_of_unique_words'] = number_unique_words

# Save the updated DataFrame to a CSV file
df.to_csv(file_path, index=False)

print(df)"""

"""import pandas as pd

file_path = "users_with_number_of_posts/year_2020.csv"
submission_file_path = "year_wise_filtered_submissions/filtered_submissions_2020.csv"
comments_file_path = "/Users/anandagarwal/reddit_mental_health_analysis/year_wise_filtered_comments/filtered_comments_2020.csv"

# Read the users DataFrame
df = pd.read_csv(file_path)

# Read the submissions and comments data
submissions_df = pd.read_csv(submission_file_path)
comments_df = pd.read_csv(comments_file_path)

# Function to concatenate text for a given author
def compile_texts(author, submissions_df, comments_df):
    submission_texts = submissions_df[submissions_df["author"] == author]["text"].tolist()
    comment_texts = comments_df[comments_df["author"] == author]["text"].tolist()
    all_texts = " ".join(submission_texts + comment_texts)
    return all_texts

# Function to count unique words in a text
def count_unique_words(text):
    words = set(text.split())
    return len(words)

# Initialize the new column in the DataFrame
df["number_of_unique_words"] = df["author"].apply(lambda author: count_unique_words(compile_texts(author, submissions_df, comments_df)))

# Save the updated DataFrame to a CSV file
df.to_csv("users_with_number_of_posts/test2.csv", index=False)

print(df)"""
import pandas as pd

file_path = "users_with_number_of_posts/year_2020.csv"
submission_file_path = "year_wise_filtered_submissions/filtered_submissions_2020.csv"

# Read the users DataFrame
df = pd.read_csv(file_path)

# Read the submissions data
submissions_df = pd.read_csv(submission_file_path)

# Print columns to check for correct column names
print("Submission DataFrame columns:", submissions_df.columns)

# Ensure that the 'text' column exists
assert 'text' in submissions_df.columns, "The 'text' column is not present in the submissions DataFrame"

# Function to concatenate text for a given author
def compile_texts(author, submissions_df):
    submission_texts = submissions_df[submissions_df["author"] == author]["text"].fillna("").astype(str).tolist()
    all_texts = " ".join(submission_texts)
    return all_texts

# Function to count unique words in a text
def count_unique_words(text):
    words = set(text.split())
    return len(words)

# Initialize the new column in the DataFrame
df["number_of_unique_words"] = df["author"].apply(
    lambda author: count_unique_words(compile_texts(author, submissions_df)) if author in submissions_df["author"].values else 0
)

# Save the updated DataFrame to a CSV file
df.to_csv(file_path, index=False)

print(df)
