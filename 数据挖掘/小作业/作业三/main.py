import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import re
np.random.seed(1234)
nltk.download('punkt')
nltk.download('stopwords')

with open('twitter.txt', 'r', encoding='utf-8') as file:
    twitters = file.readlines()

stop_words = set(stopwords.words('english'))

processed_twitters = []

for twitter in twitters:
    twitter = re.sub(r'(?:\{\@\w+\@\})|\{\{[^{}]+\}\}', '', twitter)
    tokens = nltk.word_tokenize(twitter.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    processed_twitters.append(' '.join(filtered_tokens))

# TF-IDF向量化
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(processed_twitters)

# K-means聚类
k_values = [2, 3, 4]
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=1234)
    kmeans.fit(tfidf_matrix)
    clusters = kmeans.labels_

    print(f"===== Clustering with k={k} =====")
    for cluster_id in range(k):
        cluster_twitters_indices = np.where(clusters == cluster_id)[0]
        # import ipdb; ipdb.set_trace()
        top_words_indices = tfidf_matrix[cluster_twitters_indices, :].mean(0).argsort().tolist()[0][-5:]
        top_keywords = [vectorizer.get_feature_names()[indice] for indice in top_words_indices]
        print(f"Cluster {cluster_id + 1} Keywords: {top_keywords}")