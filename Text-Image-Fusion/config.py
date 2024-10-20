import numpy as np

# data_path = './data/output_file.csv'
# data_path = './data/测试数据400_三元组.xlsx'
data_path = './img2text.xlsx'

stopwords_path = './stopwords-master/hit_stopwords.txt'

scene_only = True
if_cos = True


# next_feature_ext = 'tfidf' #input("请输入要使用的特征提取方法选项：\nA:tfidf\nB:Word2Vec\nC:Glove\nD:BOW\nF:LDA\nG:bert\n")
class next_feature_ext:
    n_cluster = 7
    ext = 'bert'  # "请输入要使用的特征提取方法选项：\nA:tfidf\nB:Word2Vec\nC:Glove\nD:BOW\nF:LDA\nG:bert\n"
    clustering = 'DBSCAN'  # 请输入要使用的聚类方法选项：\nA:DBSCAN\nB:K-means\nC:HC\n'
    file_name = ext + '_' + clustering


class tfdif:
    input = "content"  # input：string{‘filename’, ‘file’, ‘content’}
    encoding = "utf-8"  # encoding：string， ‘utf-8’by default
    decode_error = "strict"  # decode_error: {‘strict’, ‘ignore’, ‘replace’}
    strip_accents = None  # strip_accents: {‘ascii’, ‘unicode’, None}
    lowercase = True  # lowercase：boolean， default True
    preprocessor = None  # preprocessor：callable or None（default）
    tokenizer = None  # tokenizer：callable or None(default)
    analyzer = "word"  # analyzer：string，{‘word’, ‘char’} or callable
    stop_words = None  # stop_words：string {‘english’}, list, or None(default)
    token_pattern = r"(?u)\b\w\w+\b"  # token_pattern：string
    ngram_range = (1, 4)  # ngram_range: tuple(min_n, max_n)
    max_df = 0.5  # max_df： float in range [0.0, 1.0] or int, optional, 1.0 by default
    min_df = 2  # min_df：float in range [0.0, 1.0] or int, optional, 1.0 by default
    max_features = None  # max_features： optional， None by default
    vocabulary = None  # vocabulary：Mapping or iterable， optional
    binary = False  # binary：boolean， False by default
    dtype = np.float64  # dtype：type， optional
    norm = "l2"  # norm：‘l1’, ‘l2’, or None,optional
    use_idf = True  # use_idf：boolean， optional
    smooth_idf = True  # smooth_idf：boolean，optional
    sublinear_tf = True  # sublinear_tf：boolean， optional


class bow:
    input = "content"
    encoding = "utf-8"
    decode_error = "strict"
    strip_accents = None
    lowercase = True
    preprocessor = None
    tokenizer = None
    stop_words = None
    token_pattern = r"(?u)\b\w\w+\b"
    ngram_range = (1, 1)
    analyzer = "word"
    max_df = 1.0
    min_df = 1
    max_features = None
    vocabulary = None
    binary = False
    dtype = np.int64


MAX_WORDS_IN_BATCH = 10000

class word2vec:
    # sentences = None
    corpus_file = None
    vector_size = 100
    alpha = 0.025
    window = 500
    min_count = 2
    max_vocab_size = None
    sample = 1e-3
    seed = 1
    workers = 5
    min_alpha = 0.0001
    sg = 0
    hs = 0
    negative = 5
    ns_exponent = 0.75
    cbow_mean = 1
    hashfxn = hash
    epochs = 5
    null_word = 0
    trim_rule = None
    sorted_vocab = 1
    batch_words = MAX_WORDS_IN_BATCH
    compute_loss = False
    callbacks = ()
    comment = None
    max_final_vocab = None
    shrink_windows = True

class lda:
    # corpus = None
    num_topics = 10
    # id2word = None
    distributed = False
    chunksize = 2000
    passes = 1
    update_every = 1
    alpha = 'symmetric'
    eta = None
    decay = 0.5
    offset = 1.0
    eval_every = 10
    iterations = 50
    gamma_threshold = 0.001
    minimum_probability = 0.01
    random_state = 3
    ns_conf = None
    minimum_phi_value = 0.01
    per_word_topics = False
    callbacks = None
    dtype = np.float32

class dbscan:
    eps = 1.2 #0.001
    min_samples = 2
    metric = "euclidean" #欧式距离 “euclidean”,曼哈顿距离 “manhattan”,切比雪夫距离“chebyshev”,标准化欧式距离 “seuclidean”,
                        #马氏距离“mahalanobis”
    metric_params = None
    algorithm = "auto" # {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
    leaf_size = 30
    p = None
    n_jobs = None

class km: #K-Means
    # n_clusters = 8 #这个暂定和bug类型一直，可以改next_feature_ext中的n_cluster
    init = "k-means++"
    max_iter = 100
    batch_size = 1024
    verbose = 0
    compute_labels = True
    random_state = None
    tol = 0.0
    max_no_improvement = 10
    init_size = None
    n_init = 10
    reassignment_ratio = 0.01

class hc: #层次聚类
    #n_clusters = 2 #这个暂定和bug类型一直，可以改next_feature_ext中的n_cluster
    affinity = "deprecated"  # TODO(1.4): Remove
    metric = "euclidean"  # TODO(1.4): Set to "euclidean"
    memory = None
    connectivity = None
    compute_full_tree = "auto"
    linkage = "ward"
    distance_threshold = 2
    compute_distances = False