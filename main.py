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

print(x_train.shape)
print(y_train.shape)
print(x_train)
print(y_train)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#定义GP树
FeatureVector = list
Float = float

pset = PrimitiveSetTyped("MAIN", [FeatureVector], Float)
pset.renameArguments(ARG0='combined_features')

# 定义特征提取函数
def get_feature(index):
    def feature_func(features):
        return float(features[index])
    return feature_func

#添加运算符
pset.addPrimitive(lambda x, y: [x[i] + y[i] for i in range(len(x))], [FeatureVector, FeatureVector], FeatureVector, name="add")
pset.addPrimitive(lambda x, y: [x[i] - y[i] for i in range(len(x))], [FeatureVector, FeatureVector], FeatureVector, name="sub")
pset.addPrimitive(lambda x, y: [x[i] * y[i] for i in range(len(x))], [FeatureVector, FeatureVector], FeatureVector, name="mul")
pset.addPrimitive(protectedDiv, [FeatureVector, FeatureVector], FeatureVector, name="div")
pset.addPrimitive(lambda x: [np.sin(x[i]) for i in range(len(x))], [FeatureVector], FeatureVector, name="sin")
pset.addPrimitive(lambda x: [np.cos(x[i]) for i in range(len(x))], [FeatureVector], FeatureVector, name="cos")
pset.addPrimitive(fe_fs.conditional_op, [FeatureVector, FeatureVector, FeatureVector, FeatureVector], FeatureVector, name="if")
pset.addPrimitive(lambda x: np.mean(x), [FeatureVector], Float, name="mean")

# 添加特征提取终端节点 588->652
for i in range(652):
    pset.addTerminal(get_feature(i), Float, name=f"F{i}")

# 个体
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
        X_transformed = []
        for sample in x_train:
            result = func(sample.tolist())
            if callable(result):  # 处理单一特征索引
                result = result(sample.tolist())
            X_transformed.append(float(result))
        X_transformed = np.array(X_transformed).reshape(-1, 1)
        print(f"X_transformed shape: {X_transformed.shape}, first few values: {X_transformed[:5]}")

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

    X_train_transformed = np.array([
        best_func(sample.tolist()) if not callable(best_func(sample.tolist()))
        else best_func(sample.tolist())(sample.tolist())
        for sample in x_train
    ]).reshape(-1, 1)

    X_test_transformed = np.array([
        best_func(sample.tolist()) if not callable(best_func(sample.tolist()))
        else best_func(sample.tolist())(sample.tolist())
        for sample in x_test
    ]).reshape(-1, 1)

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