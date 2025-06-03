import os
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- 数据路径 ---
# 处理后的特征保存的基础路径 (split_ph2.py 的输出)
PROCESSED_FOLDER_BASE = "./processed_ph2_separate/"
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
CX_PROB = 0.80       # 交叉概率
MUT_PROB = 0.19      # 变异概率
# Elitism 需要通过 specific algorithm (如eaMuPlusLambda) 或手动实现
MU = POPULATION_SIZE      # eaMuPlusLambda: 父代数量
LAMBDA_ = POPULATION_SIZE # eaMuPlusLambda: 子代数量
TREE_MIN_DEPTH = 2   # 初始树最小深度
TREE_MAX_DEPTH = 6   # 初始树最大深度
TOURNAMENT_SIZE = 7  # 锦标赛选择大小
MAX_TREE_HEIGHT = 17 # 树最大高度限制

# --- 交叉验证参数 ---
NUM_JOBS = 5         # 重复运行次数 (ECJ 中的 jobs)
N_SPLITS = 10        # 交叉验证折数

# --- 分类器配置 ---

CLASSIFIER_CONFIG = {
    'svm': {'class': SVC, 'params': {'kernel': 'rbf', 'probability': True, 'random_state': 42, 'C': 1.0, 'gamma': 'scale'}},
    'dt':  {'class': DecisionTreeClassifier, 'params': {'criterion': 'entropy', 'random_state': 42, 'max_depth': 5}}, # 对应 J48
    'rf':  {'class': RandomForestClassifier, 'params': {'n_estimators': 10, 'random_state': 42, 'max_depth': 5, 'n_jobs': -1}},
}