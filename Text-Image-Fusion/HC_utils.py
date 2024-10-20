import pandas as pd
import jieba
from jieba import posseg
import re
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_keywords(sentences, stopwords, top_n=3):
    # 初始化TF-IDF向量化器
    vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, stop_words=list(stopwords))
    # 对每个句子进行向量化处理
    tfidf_matrix = vectorizer.fit_transform(sentences)
    # 获取特征名称
    feature_names = np.array(vectorizer.get_feature_names_out())
    # 存储每个句子的关键词
    top_keywords = []
    # 存储每个句子的TF-IDF矩阵，初始化为空数组
    tfidf_matrices = np.empty((len(sentences), tfidf_matrix.shape[1]), dtype=np.float64)
    # 遍历每个句子
    for i, text in enumerate(sentences):
        # 获取当前句子的TF-IDF数组
        tfidf_array = tfidf_matrix.toarray()[i]
        # 将TF-IDF数组赋值给tfidf_matrices的对应位置
        tfidf_matrices[i] = tfidf_array
        # 获取TF-IDF值最高的关键词的索引
        indices = tfidf_array.argsort()[-top_n:][::-1]
        # 获取关键词
        keywords = feature_names[indices]
        # 将关键词以空格分隔的形式添加到列表
        top_keywords.append(' '.join(keywords))
    return top_keywords, tfidf_matrices

# 层次聚类
def hierarchical_clustering(feature, n_clusters, distance_threshold):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold = distance_threshold, metric='euclidean', linkage='ward')
    result = clustering.fit(feature).labels_
    return result

def load_preprocess_defects(stopwords_file, defects_file):
    """
    加载缺陷数据集并进行预处理
    """
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
