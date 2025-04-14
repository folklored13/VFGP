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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# 1. 基础函数与数据预处理
def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        x = np.where(np.isfinite(x), x, 0.0)
    return x

# 加载数据
data_path = "E:/pycode/i_c/data/"
x_train = np.load(os.path.join(data_path, "HAM10000_train_features.npy"))
y_train = np.load(os.path.join(data_path, "HAM10000_train_labels.npy"))
x_test = np.load(os.path.join(data_path, "HAM10000_test_features.npy"))
y_test = np.load(os.path.join(data_path, "HAM10000_test_labels.npy"))

print(x_train.shape)  # 打印形状以确认

# 标准化处理
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2 定义GPtree
FeatureVector = list
Float = float  # 计算结果

pset = PrimitiveSetTyped("MAIN", [FeatureVector], Float)  # 输出类型改为 Float
pset.renameArguments(ARG0='combined_features')

# 定义特征提取函数
def get_feature(index):
    def feature_func(features):
        return float(features[index])  # 从特征向量中提取对应索引的值
    return feature_func

# 添加运算符
pset.addPrimitive(lambda x, y: [x[i] + y[i] for i in range(len(x))], [FeatureVector, FeatureVector], FeatureVector, name="add")
pset.addPrimitive(lambda x, y: [x[i] - y[i] for i in range(len(x))], [FeatureVector, FeatureVector], FeatureVector, name="sub")
pset.addPrimitive(lambda x, y: [x[i] * y[i] for i in range(len(x))], [FeatureVector, FeatureVector], FeatureVector, name="mul")
pset.addPrimitive(protectedDiv, [FeatureVector, FeatureVector], FeatureVector, name="div")
pset.addPrimitive(lambda x: [np.sin(x[i]) for i in range(len(x))], [FeatureVector], FeatureVector, name="sin")
pset.addPrimitive(lambda x: [np.cos(x[i]) for i in range(len(x))], [FeatureVector], FeatureVector, name="cos")
pset.addPrimitive(fe_fs.conditional_op, [FeatureVector, FeatureVector, FeatureVector, FeatureVector], FeatureVector, name="if")

# 添加标量运算符，将 FeatureVector 转换为 Float
pset.addPrimitive(lambda x: np.mean(x), [FeatureVector], Float, name="mean")

# 添加特征提取终端节点
for i in range(588):
    pset.addTerminal(get_feature(i), Float, name=f"F{i}")

# 定义GP个体
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 配置DEAP工具箱
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# 评估函数
def evalVFGP(individual):
    try:
        func = toolbox.compile(expr=individual)
        print(f"Individual: {individual}")
        print(f"Compiled function result for first sample: {func(x_train[0].tolist())}")
        X_transformed = []
        for sample in x_train:
            result = func(sample.tolist())  #
            X_transformed.append(result)
        X_transformed = np.array(X_transformed).reshape(-1, 1)

        clf1 = SVC(kernel='rbf', probability=True, random_state=42)
        clf2 = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
        eclf = VotingClassifier(estimators=[('svm', clf1), ('j48', clf2), ('rf', clf3)], voting='soft')

        f1_scores = cross_val_score(eclf, X_transformed, y_train, cv=10, scoring='f1_macro')
        avg_f1 = np.mean(f1_scores) * 100
    except Exception as e:
        print(f"Error in evalVFGP: {e}")
        avg_f1 = 0
    return (avg_f1,)

# 6. 设定GP算子
toolbox.register("evaluate", evalVFGP)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# 运行
if __name__ == "__main__":
    # 参数设置（根据论文调整）
    population_size = 20
    generations = 5
    cx_prob = 0.7
    mut_prob = 0.2
    elit_prob = 0.01

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # 运行进化
    pop, log, _ = evalGP.eaSimple(
        pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, elitpb=elit_prob, ngen=generations,
        stats=stats, halloffame=hof, verbose=True
    )

    # 测试最佳个体
    best_func = toolbox.compile(expr=hof[0])
    X_test_transformed = np.array([best_func(sample.tolist()) for sample in x_test]).reshape(-1, 1)

    eclf = VotingClassifier(
        estimators=[('svm', SVC()), ('j48', DecisionTreeClassifier()), ('rf', RandomForestClassifier())],
        voting='soft'
    )
    eclf.fit(scaler.transform(x_train), y_train)
    y_pred = eclf.predict(X_test_transformed)

    print("\n=== 最终测试结果 ===")
    print("准确率:", accuracy_score(y_test, y_pred))
    print("F1分数 (Macro):", f1_score(y_test, y_pred, average='macro'))
    print("最佳个体:", hof[0])