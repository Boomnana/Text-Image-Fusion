import json
import sys
from sklearn.base import BaseEstimator, ClusterMixin
import jieba
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.cluster import DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score
import spacy
from sklearn.metrics.cluster import contingency_matrix

from docx import Document

import config
from sentence_transformers import util, SentenceTransformer
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score

import openpyxl

stopwords_path = './stopwords-master/hit_stopwords.txt'


def purity_score(y_true, y_pred):
    # 计算纯度
    # 使用混淆矩阵（contingency matrix），其中每一列代表一个真实类别，每一行代表一个聚类
    cont_matrix = contingency_matrix(y_true, y_pred)

    # 对于每个聚类（行），找出数量最多的真实类别（列）的数量
    # 然后将这些数量加起来
    # 最后，将总数除以样本总数以计算纯度
    return np.sum(np.amax(cont_matrix, axis=0)) / np.sum(cont_matrix)

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

def hdb_cluster(data):
    clustering = HDBSCAN(min_cluster_size=2, min_samples=1).fit(data)
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

def bert_exr(sentences):
    for i in range(len(sentences)):
        sentences[i] = ' '.join(sentences[i])
    model = SentenceTransformer(
        'paraphrase-multilingual-mpnet-base-v2',  # 要使用的预训练模型
        cache_folder=r"./model"  # 指定该模型在本地的缓存路径
    )

    features = model.encode(sentences)
    features = sp.sparse.csr_matrix(features)
    return features

class EnsembleCluster(BaseEstimator, ClusterMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit_predict_mean(self, X):
        cluster_probs = []
        max_num_clusters = 0

        # 首先，我们需要找到最大的簇数量
        for estimator in self.estimators:
            if isinstance(estimator, AgglomerativeClustering):
                labels = estimator.fit_predict(X.toarray())
            else:
                labels = estimator.fit_predict(X)
            max_num_clusters = max(max_num_clusters, len(np.unique(labels)))

        # 然后，我们创建每个样本在每个簇中的概率矩阵
        for estimator in self.estimators:
            if isinstance(estimator, AgglomerativeClustering):
                labels = estimator.fit_predict(X.toarray())
            else:
                labels = estimator.fit_predict(X)
            unique_labels = np.unique(labels)
            num_clusters = len(unique_labels)

            # 创建一个填充了零的矩阵，其列数等于最大的簇数量
            probs = np.zeros((X.shape[0], max_num_clusters))

            # 使用布尔索引填充概率矩阵
            for i, label in enumerate(unique_labels):
                if label != -1:  # 不要为噪声点分配概率
                    probs[:, i] = (labels == label).astype(float)

            cluster_probs.append(probs)

        # 计算概率的平均值
        avg_probs = np.mean(cluster_probs, axis=0)

        # 为每个样本分配簇标签
        final_labels = np.argmax(avg_probs, axis=1)

        return final_labels

    def fit_predict_append(self, X):
        cluster_labels = []
        for estimator in self.estimators:
            # print(type(estimator))
            if type(estimator) is AgglomerativeClustering:
                # labels = estimator.fit(X.toarray()).labels_
                labels = estimator.fit_predict(X.toarray())
            else:
                # labels = estimator.fit(X).labels_
                labels = estimator.fit_predict(X)
            # print(type(labels))
            cluster_labels.append(labels)

        # Voting: Assign each data point to the cluster with the most votes
        ensemble_labels = np.array(cluster_labels).T

        final_labels = []
        for row in ensemble_labels:
            unique_labels, counts = np.unique(row, return_counts=True)
            if len(unique_labels) == 1 and unique_labels[0] == -1:
                # Handle samples with no cluster assignment
                final_labels.append(-1)
            else:
                final_labels.append(unique_labels[np.argmax(counts)])

        # final_labels = np.array([np.argmax(np.bincount(row)) for row in ensemble_labels])
        # print('1')

        return np.array(final_labels)

if __name__ == '__main__':
    # TODO 改一下读取文件的路径
    data_path = "缺陷报告-标注-平衡.xlsx"
    data = pd.read_excel(data_path, sheet_name="Sheet1")  # , sep=',')
    data_ = data["缺陷描述"]

    # 进行分词
    data_1 = new_cut(data_)

    # # 创建包含分词结果的DataFrame
    # df = pd.DataFrame({"分词结果": data_1})

    # # 保存处理后的数据到Excel表中
    # df.to_excel("分词结果.xlsx", index=False)

    # data_2 = [' '.join(data) for data in data_1]

    # tfidf = TfidfVectorizer()

    # TODO 特征提取

    # ti-idf
    ver_ti_idf = tfdif(data_1)

    # sentence_bert
    ver_sb = bert_exr(data_1)

    # TODO 三种聚类方法

    # HDBSCAN
    hdb_cluster_ti_idf = hdb_cluster(ver_ti_idf)
    hdb_cluster_sb = hdb_cluster(ver_sb)
    # DBSCAN
    db_cluster_ti_idf = db_cluster(ver_ti_idf)
    db_cluster_sb = db_cluster(ver_sb)
    # HC
    n = None # 聚类的个数，自己设定一下
    hc_cluster_ti_idf = hc_cluster(ver_ti_idf,n)
    hc_cluster_sb = hc_cluster(ver_sb,n)
    # TODO 还有ensemble-cluster 没有设置，这个ai说可以使用叠加，平均等
    hdbscan = HDBSCAN(min_cluster_size=2, min_samples=1)
    dbscan = DBSCAN(eps=0.03, min_samples=2)
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
    )
    # 创建集成聚类器，传入多个聚类算法
    ensemble_cluster = EnsembleCluster(estimators=[hdbscan, dbscan, hc])
    ensemble_cluster_append_ti_idf = ensemble_cluster.fit_predict_append(ver_ti_idf)
    ensemble_cluster_append_sb = ensemble_cluster.fit_predict_append(ver_sb)
    ensemble_cluster_mean_ti_idf = ensemble_cluster.fit_predict_mean(ver_ti_idf)
    ensemble_cluster_mean_sb = ensemble_cluster.fit_predict_mean(ver_sb)

    # 输出看看
    # print("ti_idf HDBSCAN Clusters:", hdb_cluster_ti_idf.labels_)
    # print("sb HDBSCAN Clusters:", hdb_cluster_sb.labels_)
    # print("ti_idf DBSCAN Clusters:", db_cluster_ti_idf.labels_)
    # print("sb DBSCAN Clusters:", db_cluster_sb.labels_)
    # print("ti_idf HC Clusters:", hc_cluster_ti_idf.labels_)
    # print("sb HC Clusters:", hc_cluster_sb.labels_)
    # print("ti_idf ensemble Clusters:", ensemble_cluster_ti_idf)
    # print("sb ensemble Clusters:", ensemble_cluster_sb)

    # TODO 指标
    y_true = data['标签']  # 这个地方从excel中提取
    # 纯度
    purity_hdb_cluster_ti_idf = purity_score(y_true, hdb_cluster_ti_idf.labels_)
    purity_hdb_cluster_sb = purity_score(y_true, hdb_cluster_sb.labels_)
    purity_db_cluster_ti_idf = purity_score(y_true, db_cluster_ti_idf.labels_)
    purity_db_cluster_sb = purity_score(y_true, hdb_cluster_sb.labels_)
    purity_hc_cluster__ti_idf = purity_score(y_true, hc_cluster_ti_idf.labels_)
    purity_hc_cluster_sb = purity_score(y_true, hc_cluster_sb.labels_)
    purity_ensemble_cluster_append__ti_idf = purity_score(y_true, ensemble_cluster_append_ti_idf)
    purity_ensemble_cluster_append_sb = purity_score(y_true, ensemble_cluster_append_sb)
    purity_ensemble_cluster_mean__ti_idf = purity_score(y_true, ensemble_cluster_mean_ti_idf)
    purity_ensemble_cluster_mean_sb = purity_score(y_true, ensemble_cluster_mean_sb)
    # 调整互信息
    ami_hdb_cluster_ti_idf = adjusted_mutual_info_score(y_true, hdb_cluster_ti_idf.labels_)
    ami_hdb_cluster_sb = adjusted_mutual_info_score(y_true, hdb_cluster_sb.labels_)
    ami_db_cluster_ti_idf = adjusted_mutual_info_score(y_true, db_cluster_ti_idf.labels_)
    ami_db_cluster_sb = adjusted_mutual_info_score(y_true, db_cluster_sb.labels_)
    ami_hc_cluster__ti_idf = adjusted_mutual_info_score(y_true, hc_cluster_ti_idf.labels_)
    ami_hc_cluster_sb = adjusted_mutual_info_score(y_true, hc_cluster_sb.labels_)
    ami_ensemble_cluster_append__ti_idf = adjusted_mutual_info_score(y_true, ensemble_cluster_append_ti_idf)
    ami_ensemble_cluster_append_sb = adjusted_mutual_info_score(y_true, ensemble_cluster_append_sb)
    ami_ensemble_cluster_mean__ti_idf = adjusted_mutual_info_score(y_true, ensemble_cluster_mean_ti_idf)
    ami_ensemble_cluster_mean_sb = adjusted_mutual_info_score(y_true, ensemble_cluster_mean_sb)
    # V-measure
    v_measure_hdb_cluster_ti_idf = v_measure_score(y_true, hdb_cluster_ti_idf.labels_)
    v_measure_hdb_cluster_sb = v_measure_score(y_true, hdb_cluster_sb.labels_)
    v_measure_db_cluster_ti_idf = v_measure_score(y_true, db_cluster_ti_idf.labels_)
    v_measure_db_cluster_sb = v_measure_score(y_true, db_cluster_sb.labels_)
    v_measure_hc_cluster__ti_idf = v_measure_score(y_true, hc_cluster_ti_idf.labels_)
    v_measure_hc_cluster_sb = v_measure_score(y_true, hc_cluster_sb.labels_)
    v_measure_ensemble_cluster_append__ti_idf = v_measure_score(y_true, ensemble_cluster_append_ti_idf)
    v_measure_ensemble_cluster_append_sb = v_measure_score(y_true, ensemble_cluster_append_sb)
    v_measure_ensemble_cluster_mean__ti_idf = v_measure_score(y_true, ensemble_cluster_mean_ti_idf)
    v_measure_ensemble_cluster_mean_sb = v_measure_score(y_true, ensemble_cluster_mean_sb)
    # 输出指标看下
    # 输出纯度指标
    print('======================纯度======================')
    print("Purity HDBSCAN Clusters (ti_idf):", purity_hdb_cluster_ti_idf)
    print("Purity HDBSCAN Clusters (sb):", purity_hdb_cluster_sb)
    print("Purity DBSCAN Clusters (ti_idf):", purity_db_cluster_ti_idf)
    print("Purity DBSCAN Clusters (sb):", purity_db_cluster_sb)
    print("Purity HC Clusters (ti_idf):", purity_hc_cluster__ti_idf)
    print("Purity HC Clusters (sb):", purity_hc_cluster_sb)
    print("Purity ensemble Clusters append (ti_idf):", purity_ensemble_cluster_append__ti_idf)
    print("Purity ensemble Clusters append (sb):", purity_ensemble_cluster_append_sb)
    print("Purity ensemble Clusters mean (ti_idf):", purity_ensemble_cluster_mean__ti_idf)
    print("Purity ensemble Clusters mean (sb):", purity_ensemble_cluster_mean_sb)
    # 输出调整后的互信息得分
    print('====================调整后互信息====================')
    print("Adjusted Mutual Info Score HDBSCAN Clusters (ti_idf):", ami_hdb_cluster_ti_idf)
    print("Adjusted Mutual Info Score HDBSCAN Clusters (sb):", ami_hdb_cluster_sb)
    print("Adjusted Mutual Info Score DBSCAN Clusters (ti_idf):", ami_db_cluster_ti_idf)
    print("Adjusted Mutual Info Score DBSCAN Clusters (sb):", ami_db_cluster_sb)
    print("Adjusted Mutual Info Score HC Clusters (ti_idf):", ami_hc_cluster__ti_idf)
    print("Adjusted Mutual Info Score HC Clusters (sb):", ami_hc_cluster_sb)
    print("Adjusted Mutual Info Score ensemble Clusters append (ti_idf):", ami_ensemble_cluster_append__ti_idf)
    print("Adjusted Mutual Info Score ensemble Clusters append (sb):", ami_ensemble_cluster_append_sb)
    print("Adjusted Mutual Info Score ensemble Clusters mean (ti_idf):", ami_ensemble_cluster_mean__ti_idf)
    print("Adjusted Mutual Info Score ensemble Clusters mean (sb):", ami_ensemble_cluster_mean_sb)
    # 输出V-measure得分
    print('=====================V-measure====================')
    print("V-measure Score HDBSCAN Clusters (ti_idf):", v_measure_hdb_cluster_ti_idf)
    print("V-measure Score HDBSCAN Clusters (sb):", v_measure_hdb_cluster_sb)
    print("V-measure Score DBSCAN Clusters (ti_idf):", v_measure_db_cluster_ti_idf)
    print("V-measure Score DBSCAN Clusters (sb):", v_measure_db_cluster_sb)
    print("V-measure Score HC Clusters (ti_idf):", v_measure_hc_cluster__ti_idf)
    print("V-measure Score HC Clusters (sb):", v_measure_hc_cluster_sb)
    print("V-measure Score ensemble Clusters append (ti_idf):", v_measure_ensemble_cluster_append__ti_idf)
    print("V-measure Score ensemble Clusters append (sb):", v_measure_ensemble_cluster_append_sb)
    print("V-measure Score ensemble Clusters mean (ti_idf):", v_measure_ensemble_cluster_mean__ti_idf)
    print("V-measure Score ensemble Clusters mean (sb):", v_measure_ensemble_cluster_mean_sb)
