import csv
from datetime import datetime

input_file = r"/Users/anandagarwal/Downloads/reddit/subreddits23/filtered_submissions_5years.csv"
output_file = r"/Users/anandagarwal/Downloads/reddit/subreddits23/filtered_submissions_2023.csv"

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
    reader = csv.reader(fin, delimiter=',')
    writer = csv.writer(fout, delimiter=',')
    
    
    header = next(reader)
    writer.writerow(header)
    
    
    author_index = header.index('author')
    date_index = header.index('created')  

    for row in reader:
        row_date = datetime.strptime(row[date_index], '%Y-%m-%d %H:%M')  
            
        if row[author_index] != 'u/[deleted]' and start_date <= row_date <= end_date:
            writer.writerow(row)
