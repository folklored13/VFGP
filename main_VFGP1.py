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
# 定义保护性除法
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

print(x_train.shape)

# 标准化处理
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 定义GP语法树
FeatureVector = list  # 特征向量类型
pset = PrimitiveSetTyped("MAIN", [FeatureVector], FeatureVector)
pset.renameArguments(ARG0='combined_features')

# 添加运算符和终端节点
pset.addPrimitive(np.add, [FeatureVector, FeatureVector], FeatureVector, name="add")
pset.addPrimitive(np.subtract, [FeatureVector, FeatureVector], FeatureVector, name="sub")
pset.addPrimitive(np.multiply, [FeatureVector, FeatureVector], FeatureVector, name="mul")
pset.addPrimitive(protectedDiv, [FeatureVector, FeatureVector], FeatureVector, name="div")
pset.addPrimitive(np.sin, [FeatureVector], FeatureVector, name="sin")
pset.addPrimitive(np.cos, [FeatureVector], FeatureVector, name="cos")
pset.addPrimitive(fe_fs.conditional_op, [FeatureVector, FeatureVector, FeatureVector, FeatureVector], FeatureVector, name="if")

# 添加特征索引作为终端
for i in range(588):  # 根据论文总特征数调整
    pset.addTerminal(i, int, name=f"F{i}")

# 3. 定义GP个体
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 4. 配置DEAP工具箱
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# 5. 适应度评估函数
def evalVFGP(individual):
    try:
        func = toolbox.compile(expr=individual)
        # 对每个样本计算特征值
        X_transformed = []
        for sample in x_train:
            # 将特征向量传入 func，确保终端节点返回特征值
            result = func(sample.tolist())  # 转换为列表以匹配 FeatureVector 类型
            if isinstance(result, (list, np.ndarray)):  # 如果返回的是向量，取第一个值或平均值
                result = np.mean(result) if len(result) > 0 else 0
            X_transformed.append(result)
        X_transformed = np.array(X_transformed).reshape(-1, 1)

        # 使用集成分类器
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
toolbox.register("selectElitism",tools.selBest) # 添加精英选择
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# 7. 运行GP进化
if __name__ == "__main__":
    # 参数设置（根据论文调整）
    population_size = 100
    generations = 50
    cx_prob = 0.7
    mut_prob = 0.2
    elit_prob = 0.01  # 添加精英选择比例，根据table4

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # 运行进化
    pop, log, best_ind_gen = evalGP.eaSimple(
        pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, elitpb=elit_prob, ngen=generations,
        stats=stats, halloffame=hof, verbose=True
    )

    # 测试最佳个体
    best_func = toolbox.compile(expr=hof[0])
    X_test_transformed = np.array([best_func(sample) for sample in x_test]).reshape(-1, 1)

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