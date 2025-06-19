import os
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- 数据路径 ---
# 处理后的特征保存的基础路径 (split_ph2.py 的输出)
PROCESSED_RUNS_BASE = "E:/datasets/PH2-dataset/processed_ph2_runs/"
# 根据 feature_type 会自动添加子目录名 (e.g., "./processed_ph2_separate/lbpgray")

# --- 特征维度 ---
FEATURE_DIMS = {
    'lbpgray': 59,
    'lbprgb': 177,
    'wavelet': 416
}

# --- GP 参数 ---
POPULATION_SIZE = 100
GENERATIONS = 50
CX_PROB = 0.80
MUT_PROB = 0.19
# Elitism 需要通过 specific algorithm (如eaMuPlusLambda)
MU = POPULATION_SIZE      # 父代数量
LAMBDA_ = POPULATION_SIZE # 子代数量
TREE_MIN_DEPTH = 2   # 初始树最小深度
TREE_MAX_DEPTH = 6   # 初始树最大深度
TOURNAMENT_SIZE = 6  # 锦标赛选择大小7改为6



NUM_JOBS = 5         # 重复运行次数jobs
N_SPLITS = 10


CLASSIFIER_CONFIG = {
    'svm': {'class': SVC, 'params': {'kernel': 'rbf', 'probability': True, 'random_state': 42, 'C': 100, 'gamma': 0.1}},#将 SVM 的参数设置为 C=100 和 gamma=0.1
    'dt':  {'class': DecisionTreeClassifier, 'params': {'criterion': 'entropy', 'random_state': 42, 'max_depth': 5}}, # 对应 J48
    'rf':  {'class': RandomForestClassifier, 'params': {'n_estimators': 5, 'random_state': 42, 'max_depth': 5, 'n_jobs': -1}},#使用的树数量是 5
}

CLASS_MAP = {
    'Common Nevus': 0,
    'Atypical Nevus': 1,
    'Melanoma': 2
}