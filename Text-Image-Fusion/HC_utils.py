import pandas as pd
import jieba
from jieba import posseg
import re
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_keywords(sentences, stopwords, top_n=3):

    vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, stop_words=list(stopwords))

    tfidf_matrix = vectorizer.fit_transform(sentences)

    feature_names = np.array(vectorizer.get_feature_names_out())

    top_keywords = []

    tfidf_matrices = np.empty((len(sentences), tfidf_matrix.shape[1]), dtype=np.float64)

    for i, text in enumerate(sentences):

        tfidf_array = tfidf_matrix.toarray()[i]

        tfidf_matrices[i] = tfidf_array

        indices = tfidf_array.argsort()[-top_n:][::-1]

        keywords = feature_names[indices]

        top_keywords.append(' '.join(keywords))
    return top_keywords, tfidf_matrices


def hierarchical_clustering(feature, n_clusters, distance_threshold):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold = distance_threshold, metric='euclidean', linkage='ward')
    result = clustering.fit(feature).labels_
    return result

def load_preprocess_defects(stopwords_file, defects_file):

    df = pd.read_excel(defects_file)
    with open(stopwords_file, 'r') as f:
        stopwords = set(f.read().splitlines())

    def preprocess_text(text):
        words = jieba.lcut(text)
        words = [word for word in words if word not in stopwords]
        return ' '.join(words)

    df['processed_text'] = df['缺陷描述'].apply(preprocess_text)
    return df

def save_results(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False,encoding='utf-8-sig')
    print(f"Results saved to {output_file}")
