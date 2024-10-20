import json
import sys

import jieba
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.cluster import DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score
import spacy

from docx import Document

import config
from sentence_transformers import util

import openpyxl

stopwords_path = './stopwords-master/hit_stopwords.txt'


def cos_(ver,data_1):
    data_1 = data_1.tolist()
    cos_sims = []
    ver = ver.toarray()
    for j in range(len(ver)):
        rows = []
        for i in range(len(ver)):
            sim = util.cos_sim(ver[j],ver[i])
            rows.append(sim.tolist()[0][0])
        cos_sims.append(rows)

    #存储成Excel
    workbook = openpyxl.Workbook()
    worksheet = workbook.active

    worksheet.append(['']+data_1)
    for data,val in zip(data_1,cos_sims):
        worksheet.append([data]+val)

    workbook.save(config.next_feature_ext.ext+".xlsx")
    workbook.close()

    return sp.sparse.csr_matrix(cos_sims)

#  分词以及去除停用词
def new_cut(data):
    data = data.apply(lambda x: ' '.join(jieba.lcut(x)))
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]
    # stop_words += (['\n', ' ', '_x000D_', '摘要', '进行', '进行了', '操作', 'bug', '出现', '\\', 'n'])
    stop_words.append('\n')
    stop_words.append(' ')
    stop_words.append('_x000D_')
    # stop_words.append('摘要')
    sentences = []
    for i in range(len(data)):
        word = list(jieba.cut(data[i]))
        word = [word1 for word1 in word if not word1 in stop_words]
        # data[i] = ' '.join(word)
        sentences.append(word)
    return sentences


def tfdif(data):
    for i in range(len(data)):
        data[i] = ' '.join(data[i])
    vectorizer_word = TfidfVectorizer(
        input=config.tfdif.input,
        encoding=config.tfdif.encoding,
        decode_error=config.tfdif.decode_error,
        strip_accents=config.tfdif.strip_accents,
        lowercase=config.tfdif.lowercase,
        preprocessor=config.tfdif.preprocessor,
        tokenizer=config.tfdif.tokenizer,
        analyzer=config.tfdif.analyzer,
        stop_words=config.tfdif.stop_words,
        token_pattern=config.tfdif.token_pattern,
        ngram_range=config.tfdif.ngram_range,
        max_df=config.tfdif.max_df,
        min_df=config.tfdif.min_df,
        max_features=config.tfdif.max_features,
        vocabulary=config.tfdif.vocabulary,
        binary=config.tfdif.binary,
        dtype=config.tfdif.dtype,
        norm=config.tfdif.norm,
        use_idf=config.tfdif.use_idf,
        smooth_idf=config.tfdif.smooth_idf,
        sublinear_tf=config.tfdif.sublinear_tf,
    )
    # vectorizer_word = TfidfVectorizer(
    #     max_features=800000,
    #                                   token_pattern=r"(?u)\b\w+\b",
    #                                   min_df=5,
    #                                   # max_df=0.1,
    #                                   analyzer='word',
    #                                   ngram_range=(1, 2)
    #                                   )
    vectorizer_word = vectorizer_word.fit(data)
    # print(type(data))
    # print(vectorizer_word)
    tfidf_matrix = vectorizer_word.transform(data)
    return tfidf_matrix

def db_cluster(data):
    # clustering = DBSCAN(
    #     eps=config.dbscan.eps,
    #     min_samples=config.dbscan.min_samples,
    #     metric=config.dbscan.metric,
    #     metric_params=config.dbscan.metric_params,
    #     algorithm=config.dbscan.algorithm,
    #     leaf_size=config.dbscan.leaf_size,
    #     p=config.dbscan.p,
    #     n_jobs=config.dbscan.n_jobs,
    # ).fit(data)
    clustering = DBSCAN(eps=0.03, min_samples=2).fit(data)
    # clustering = DBSCAN(eps=0.85, min_samples=2).fit(data)
    return clustering

def npeMethod(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

def data_post(data, cluster, file_name):
    data['labels_'] = cluster.labels_

    # 存储为json
    json_data = {}
    for i in range(max(cluster.labels_) + 1):
        json_data[f"Cluster {i + 1}"] = list(data[data['labels_'] == i]['discription'])
    json_data[f"Noise:"] = list(data[data['labels_'] == -1]['discription'])

    # file_name = '' + next_feature_ext + '_' + next_cluster_me + '.json'
    with open('./result/' + file_name+".json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_data, ensure_ascii=False, indent=2))
    # json.dump(json_data,f)

def hc_cluster(data, n):
    hc = AgglomerativeClustering(
        n_clusters=n,
        affinity=config.hc.affinity,  # TODO(1.4): Remove
        metric=config.hc.metric,  # TODO(1.4): Set to "euclidean"
        memory=config.hc.memory,
        connectivity=config.hc.connectivity,
        compute_full_tree=config.hc.compute_full_tree,
        linkage=config.hc.linkage,
        distance_threshold=config.hc.distance_threshold,
        compute_distances=config.hc.compute_distances,
    ).fit(data.toarray())
    # hc = AgglomerativeClustering(n_clusters=n, linkage='average').fit(data.toarray())
    return hc

if __name__ == '__main__':

    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, HDBSCAN
    from sklearn.base import BaseEstimator, ClusterMixin
    import numpy as np

    # 生成随机数据集
    X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=1.0, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(type(X))
    sys.exit()


    class EnsembleCluster(BaseEstimator, ClusterMixin):
        def __init__(self, estimators):
            self.estimators = estimators

        def fit_predict(self, X):
            cluster_labels = []
            for estimator in self.estimators:
                labels = estimator.fit_predict(X)
                cluster_labels.append(labels)

            # Voting: Assign each data point to the cluster with the most votes
            ensemble_labels = np.array(cluster_labels).T
            final_labels = np.array([np.argmax(np.bincount(row)) for row in ensemble_labels])

            return final_labels


    # 创建多个聚类算法作为基学习器
    hdbscan = HDBSCAN(min_cluster_size=5, min_samples=5)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    hierarchical = AgglomerativeClustering(n_clusters=3)

    # 创建集成聚类器，传入多个聚类算法
    ensemble_cluster = EnsembleCluster(estimators=[hdbscan, dbscan, hierarchical])

    # 使用集成聚类器进行聚类
    predicted_labels = ensemble_cluster.fit_predict(X)

    print(predicted_labels)

    sys.exit()
    # TODO 改一下读取文件的路径
    data_path = "缺陷报告-标注-平衡.xlsx"
    data = pd.read_excel(data_path, sheet_name="Sheet1")  # , sep=',')
    data_ = data["缺陷描述"]

    # 进行分词
    data_1 = new_cut(data_)

    # 创建包含分词结果的DataFrame
    df = pd.DataFrame({"分词结果": data_1})

    # 保存处理后的数据到Excel表中
    df.to_excel("分词结果.xlsx", index=False)

    data_2 = [' '.join(data) for data in data_1]

    tfidf = TfidfVectorizer()

    # 计算TF-IDF值
    tfidf_matrix = tfidf.fit_transform(data_2)
    # TODO 还需要有sentence-bert的特征提取方式

    # 获取词汇表
    feature_names = tfidf.get_feature_names_out()

    # 找到每个句子中的关键词
    keywords = []
    for i, sentence in enumerate(data_2):
        feature_index = tfidf_matrix[i, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        top_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:3]  # 选取前3个关键词
        keywords.append([feature_names[i] for i, _ in top_keywords])

    # 创建DataFrame并保存到Excel
    df_ = pd.DataFrame({'Sentence': data_2, 'Keywords': keywords})
    df_.to_excel('keywords_output.xlsx', index=False)
    df["Sentence"] = data_2
    df["Keywords"] = keywords

    ver = tfdif(data_1)

    if_cos = False
    if if_cos:
        cos_sims = util.cos_(ver, data_)
        ver = cos_sims

    # TODO 聚类方式要有多个
    cluster_db = db_cluster(ver)
    df["db聚类"] = cluster_db.labels_

    # 聚类的个数，自己设定一下
    n = 10
    cluster_hc = hc_cluster(ver, 10)
    df["hc聚类"] = cluster_db.labels_


    df.to_excel("result.xlsx", index=False)