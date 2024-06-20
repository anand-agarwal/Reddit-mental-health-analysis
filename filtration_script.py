import pandas as pd

df=pd.read_csv("/Users/anandagarwal/reddit_mental_health_analysis/users_with_number_of_posts/year_2020.csv")
x=df.drop(columns=["number_of_unique_words_x","number_of_unique_words_y","number_of_unique_words"])
print(x)
x.to_csv("/Users/anandagarwal/reddit_mental_health_analysis/users_with_number_of_posts/year_2020.csv")

"""# Read the CSV files
comments_df = pd.read_csv("/Users/anandagarwal/reddit_mental_health_analysis/year_wise_filtered_comments/filtered_comments_2021.csv")
users_df = pd.read_csv("/Users/anandagarwal/reddit_mental_health_analysis/users_with_number_of_posts/year_2021.csv")

# Get the counts of comments per author
users_df=users_df.rename(columns={"count": "number_of_posts"})
values = comments_df['author'].value_counts()

# Initialize a new column in users_df to store the number of comments
users_df['number_of_comments'] = 0

# Update the number of comments for existing users
for i in range(len(users_df)):
    user = users_df.at[i, 'author']
    if user in values.index:
        users_df.at[i, 'number_of_comments'] = values[user]

# Collect new authors that are not in users_df
new_rows = []
for author in values.index:
    if author not in users_df['author'].values:
        new_row = {'author': author, "number_of_posts":0, 'number_of_comments': values[author]}
        new_rows.append(new_row)

# Concatenate the new rows to the users_df DataFrame
if new_rows:
    new_df = pd.DataFrame(new_rows)
    users_df = pd.concat([users_df, new_df], ignore_index=True)

# Save the updated DataFrame to a new CSV file
users_df.to_csv("/Users/anandagarwal/reddit_mental_health_analysis/users_with_number_of_posts/year_2020.csv", index=False)

print(users_df)"""