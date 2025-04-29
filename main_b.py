import random
import numpy as np
import os
from deap import base, creator, tools, gp
from deap.gp import PrimitiveSetTyped
import evalGP_fgp as evalGP
import fgp_functions as fe_fs
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from functools import partial

def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        x = np.where(np.isfinite(x), x, 0.0)
    return x

# 加载数据
data_path = "E:/datasets/PH2-dataset/processed_ph2/"
x_train = np.load(os.path.join(data_path, "train_features.npy"))
y_train = np.load(os.path.join(data_path, "train_labels.npy"))
x_test = np.load(os.path.join(data_path, "test_features.npy"))
y_test = np.load(os.path.join(data_path, "test_labels.npy"))

# print(x_train.shape)
# print(y_train.shape)
# print(x_train)
# print(y_train)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#定义GP树
FeatureVector = list
Float = float

pset = gp.PrimitiveSetTyped("MAIN", [list], list, "combined_features")
pset.renameArguments(ARG0='combined_features')

def vec_add(a, b):
    if not isinstance(a, (list, np.ndarray)):
        a = a([0] * 652)
    if not isinstance(b, (list, np.ndarray)):
        b = b([0] * 652)
    return [a[i] + b[i] for i in range(len(a))]
def vec_sub(a, b):
    """向量减法：逐元素相减"""
    if not isinstance(a, (list, np.ndarray)):
        a = a([0] * 652)
    if not isinstance(b, (list, np.ndarray)):
        b = b([0] * 652)
    return [a[i] - b[i] for i in range(len(a))]

def vec_mul(a, b):
    """向量乘法：逐元素相乘"""
    if not isinstance(a, (list, np.ndarray)):
        a = a([0] * 652)
    if not isinstance(b, (list, np.ndarray)):
        b = b([0] * 652)
    return [a[i] * b[i] for i in range(len(a))]

def vec_div(a, b):
    """保护性向量除法：避免除零"""
    if not isinstance(a, (list, np.ndarray)):
        a = a([0] * 652)
    if not isinstance(b, (list, np.ndarray)):
        b = b([0] * 652)
    result = []
    for ai, bi in zip(a, b):
        if bi == 0:
            result.append(0.0)
        else:
            result.append(ai / bi)
    return result

def vec_sin(a):
    """向量正弦：逐元素计算"""
    if not isinstance(a, (list, np.ndarray)):
        a = a([0] * 652)
    return [np.sin(float(x)) for x in a]

def vec_cos(a):
    """向量余弦：逐元素计算"""
    if not isinstance(a, (list, np.ndarray)):
        a = a([0] * 652)  # 提供默认输入确保可调用
    return [np.cos(float(x)) for x in a]

def vec_if(a, b, c, d):
    """向量条件操作：a < b ? c : d"""
    if not isinstance(a, (list, np.ndarray)):
        a = a([0] * 652)
    if not isinstance(b, (list, np.ndarray)):
        b = b([0] * 652)
    if not isinstance(c, (list, np.ndarray)):
        c = c([0] * 652)
    if not isinstance(d, (list, np.ndarray)):
        d = d([0] * 652)
    return [c[i] if a[i] < b[i] else d[i] for i in range(len(a))]

def vec_mean(a):
    """向量均值：返回标量（需包装为列表以保持类型一致）"""
    return [np.mean(a)]

def vec_concat(a, b):
    """向量拼接：合并两个特征向量"""
    return a + b
#添加运算符
pset.addPrimitive(vec_add, [list, list], list, name="add")
pset.addPrimitive(vec_sub, [list, list], list, name="sub")
pset.addPrimitive(vec_mul, [list, list], list, name="mul")
pset.addPrimitive(vec_div, [list, list], list, name="div")

# 添加三角函数
pset.addPrimitive(vec_sin, [list], list, name="sin")
pset.addPrimitive(vec_cos, [list], list, name="cos")

# 添加条件操作符
pset.addPrimitive(vec_if, [list, list, list, list], list, name="If")

# 添加统计操作符（可选）
# pset.addPrimitive(vec_mean, [list], list, name="mean")

# 添加特征拼接操作符（用于构造多维特征）
# pset.addPrimitive(vec_concat, [list, list], list, name="concat")

class FeatureSelector:
    def __init__(self, index):
        self.index = index

    def __call__(self, features):
        try:
            # 确保返回的是包含单个特征的列表
            return [float(features[self.index])]
        except (IndexError, TypeError) as e:
            print(f"Feature selection error at index {self.index}: {e}")
            return [0.0]  # 返回默认值

    def __repr__(self):
        return f"F{self.index}"

# 添加终端节点
for i in range(652):
    selector = FeatureSelector(i)
    pset.addTerminal(selector, list, name=f"F{i}")
    pset.context[f"F{i}"] = selector
# 添加特征提取终端节点
# for i in range(652):
#     #pset.addTerminal(i, name=f"F{i}")
#     pset.addEphemeralConstant(f"F{i}", lambda :i)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalVFGP(individual):
    try:
        func = toolbox.compile(expr=individual)
        print(f"Individual: {individual}")
        # # 1. 获取选中的特征索引（从终端节点提取）
        # selected_features = [node.value for node in individual if isinstance(node, gp.Terminal)]
        #
        # # 2. 生成构造特征
        # constructed_feature = np.array([func for _ in x_train])  # 注意：func现在无参数
        #
        # # 3. 合并特征
        # X_transformed = np.hstack([x_train[:, selected_features], constructed_feature.reshape(-1, 1)])

        # 获取选中的特征索引（从终端节点提取整数）
        selected_features = []
        for node in individual:
            if isinstance(node, gp.Terminal) and not isinstance(node.value, str):
                selected_features.append(node.value)  # 直接获取int类型的特征索引

        # 检查是否选中至少一个特征
        if len(selected_features) == 0:
            return (0,)

        # 生成构造特征（无需参数）
        constructed_feature = np.array([func(feat) for feat in x_train])
        if constructed_feature.ndim == 1:
            constructed_feature = constructed_feature.reshape(-1, 1)

        # 合并特征（确保selected_features是整数列表）
        X_transformed = np.hstack([x_train[:, selected_features], constructed_feature.reshape(-1, 1)])

        clf1 = SVC(kernel='rbf', probability=True, random_state=42)
        clf2 = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
        eclf = VotingClassifier(estimators=[('svm', clf1), ('j48', clf2), ('rf', clf3)], voting='soft')

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        f1_scores = []
        for train_idx, val_idx in skf.split(X_transformed, y_train):
            # 分割数据
            X_train_fold, X_val_fold = X_transformed[train_idx], X_transformed[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # 训练并预测
            eclf.fit(X_train_fold, y_train_fold)
            y_pred = eclf.predict(X_val_fold)

            # 计算每个类别的F1后取平均
            f1_per_class = f1_score(y_val_fold, y_pred, average=None)
            f1_scores.append(np.mean(f1_per_class))

        avg_f1 = np.mean(f1_scores) * 100

    except Exception as e:
        print(f"Error in evalVFGP: {e}")
        avg_f1 = 0
    return (avg_f1,)


toolbox.register("evaluate", evalVFGP)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


if __name__ == "__main__":
    population_size = 100
    generations = 50
    cx_prob = 0.8
    mut_prob = 0.19
    elit_prob = 0.01

    # # 测试操作符
    # a = [1.0, 2.0, 3.0]
    # b = [4.0, 5.0, 6.0]
    #
    # print("add:", vec_add(a, b))  # 应输出 [5.0, 7.0, 9.0]
    # print("div:", vec_div(a, b))  # 应输出 [0.25, 0.4, 0.5]
    # print("sin:", vec_sin(a))  # 应输出 [np.sin(1.0), np.sin(2.0), np.sin(3.0)]
    # print("concat:", vec_concat(a, b))

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log, _ = evalGP.eaSimple(
        pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, elitpb=elit_prob, ngen=generations,
        stats=stats, halloffame=hof, verbose=True
    )

    best_func = toolbox.compile(expr=hof[0])

    # X_train_transformed = np.array([best_func for _ in x_train]).reshape(-1, 1)
    #
    # X_test_transformed = np.array([best_func for _ in x_test]).reshape(-1, 1)

    selected_features = [node.value for node in hof[0] if
                         isinstance(node, gp.Terminal) and not isinstance(node.value, str)]
    X_train_transformed = np.hstack([
        x_train[:, selected_features],
        np.array([best_func(feat) for feat in x_train]).reshape(-1, 1)
    ])
    X_test_transformed = np.hstack([
        x_test[:, selected_features],
        np.array([best_func(feat) for feat in x_test]).reshape(-1, 1)
    ])

    eclf = VotingClassifier(
        estimators=[
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('j48', DecisionTreeClassifier(criterion='entropy', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ],
        voting='soft'
    )
    eclf.fit(X_train_transformed, y_train)  # 变换后的特征
    y_pred = eclf.predict(X_test_transformed)

    print("\n=== 最终测试结果 ===")
    print("准确率:", accuracy_score(y_test, y_pred))
    print("F1分数 (Macro):", f1_score(y_test, y_pred, average='macro'))
    print("最佳个体:", hof[0])

    print("\n=== 详细评估 ===")

    # 使用完整训练集训练最佳模型
    eclf.fit(X_train_transformed, y_train)

    # 训练集性能
    y_train_pred = eclf.predict(X_train_transformed)
    print("训练集准确率:", accuracy_score(y_train, y_train_pred))
    print("训练集F1分数:", f1_score(y_train, y_train_pred, average='macro'))

    # 测试集性能
    y_test_pred = eclf.predict(X_test_transformed)
    print("测试集准确率:", accuracy_score(y_test, y_test_pred))
    print("测试集F1分数:", f1_score(y_test, y_test_pred, average='macro'))

    # 输出最佳个体使用的特征
    print("\n使用的原始特征索引:", selected_features)
    print("构造的特征表达式:", hof[0])

