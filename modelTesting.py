import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
filePath = "/Users/farzanfaisal/Desktop/musicRecommender/dataSet/spotify_millsongdata.csv"
df = pd.read_csv(filePath)

# Sample 10,000 rows from the DataFrame, drop 'link' column, and reset index
df = df.sample(2000).drop('link', axis=1).reset_index(drop=True)

# TEXT Preprocessing
df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ', regex=True).replace(r'\n', ' ', regex=True)


# Tokenization using SnowballStemmer
def token(text):
    stemmer = SnowballStemmer("english")
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed_tokens)


df['text'] = df['text'].apply(token)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidf_vectorizer.fit_transform(df['text'])

# Calculate cosine similarity matrix
similar = cosine_similarity(matrix)


# Function for Recommendations
def recommenderForSong(song_name):
    idx = df[df['song'] == song_name].index
    if not idx.empty:
        idx = idx[0]  # Take the first element if there are multiple
        distance = sorted(enumerate(similar[idx]), reverse=True, key=lambda x: x[1])

        recommended_songs = []
        for s_id in distance[1:5]:
            recommended_songs.append(df.iloc[s_id[0]].song)

        return recommended_songs
    else:
        #print(f"The song '{song_name}' is not found in the DataFrame.")
        return []


# Example usage
song_to_recommend = input("Enter Song Name: ")
recommended_songs = recommenderForSong(song_to_recommend)
print(f"Recommended songs for '{song_to_recommend}': {recommended_songs}")

import pickle
pickle.dump(similar, open("similarity.pkl","wb"))
pickle.dump(df,open("df.pkl","wb"))






