import nltk
nltk.download("stopwords")



# from wordcloud import WordCloud, STOPWORDS
# import pandas as pd
# import matplotlib.pyplot as plt 

# file_path='/Users/anandagarwal/reddit_mental_health_analysis/Clusters_Centers_text/2023/cluster_0.csv'

# df=pd.read_csv(file_path)
# words=""
# stopwords = set(STOPWORDS)

# for  text in df.text:
#     text=str(text)
#     tokens=text.split()
    
#     for i in range(len(tokens)):
        
#         tokens[i]=tokens[i].lower()
        
#     words+=" ".join(tokens)+" "

# wordcloud=WordCloud(width = 800, height = 800,
#                 background_color ='white',
#                 stopwords = stopwords,
#                 min_font_size = 10).generate(words)

# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 0)
 
# plt.show()

    