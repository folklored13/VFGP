import random
import numpy as np
import os
from deap import base, creator, tools, gp
from deap.gp import PrimitiveSetTyped
import evalGP_fgp as evalGP  # 假设此模块与论文一致
import fgp_functions as fe_fs  # 假设此模块提供条件操作
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


# 保护除法，确保除以零或无效运算返回0.0
def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        x = np.where(np.isfinite(x), x, 0.0)  # 确保所有非有限值返回0.0
    return x


# 加载数据
data_path = "E:/datasets/PH2-dataset/processed_ph2/"
x_train = np.load(os.path.join(data_path, "train_features.npy"))
y_train = np.load(os.path.join(data_path, "train_labels.npy"))
x_test = np.load(os.path.join(data_path, "test_features.npy"))
y_test = np.load(os.path.join(data_path, "test_labels.npy"))

print(f"训练特征形状: {x_train.shape}")
print(f"训练标签形状: {y_train.shape}")
print(f"测试特征形状: {x_test.shape}")
print(f"测试标签形状: {y_test.shape}")

# 特征标准化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 定义GP树
pset = PrimitiveSetTyped("MAIN", [], float)  # 输出为浮点数，无输入参数

# 添加运算符
pset.addPrimitive(lambda x, y: x + y, [float, float], float, name="add")
pset.addPrimitive(lambda x, y: x - y, [float, float], float, name="sub")
pset.addPrimitive(lambda x, y: x * y, [float, float], float, name="mul")
pset.addPrimitive(protectedDiv, [float, float], float, name="div")
pset.addPrimitive(np.sin, [float], float, name="sin")
pset.addPrimitive(np.cos, [float], float, name="cos")
pset.addPrimitive(fe_fs.conditional_op, [float, float, float, float], float, name="if")  # 假设条件操作接受浮点数

# 添加特征终端节点，明确区分特征类型
feature_types = {
    'LBP_Gray': (0, 59),  # 0到58
    'LBP_RGB': (59, 236),  # 59到235
    'Wavelet': (236, 652)  # 236到651
}

for f_type, (start, end) in feature_types.items():
    for i in range(start, end):
        pset.addTerminal(lambda x, idx=i: float(x[idx]), float, name=f"{f_type}_F{i}")

# 验证终端节点
print(f"终端节点数量: {sum(len(pset.terminals[typ]) for typ in pset.terminals)}")
print(f"float 类型终端节点: {len(pset.terminals[float])}")

# 定义个体和适应度
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# 提取GP树中的终端特征索引
def get_selected_features(individual):
    terminals = [node.name for node in individual if isinstance(node, gp.Terminal)]
    feature_indices = []
    for term in terminals:
        if term.startswith(('LBP_Gray_F', 'LBP_RGB_F', 'Wavelet_F')):
            idx = int(term.split('_F')[-1])
            feature_indices.append(idx)
    return sorted(list(set(feature_indices)))  # 去重并排序


# 评估函数
def evalVFGP(individual):
    try:
        func = toolbox.compile(expr=individual)
        selected_features = get_selected_features(individual)
        X_transformed = []

        # 为每个样本生成特征向量（选定特征 + 构造特征）
        for sample in x_train:
            # 计算构造特征
            constructed_feature = func(*[float(sample[i]) for i in range(x_train.shape[1])])
            # 拼接选定特征和构造特征
            feature_vector = [float(sample[i]) for i in selected_features] + [float(constructed_feature)]
            X_transformed.append(feature_vector)

        X_transformed = np.array(X_transformed)
        print(f"个体: {individual}")
        print(f"选定特征索引: {selected_features}")
        print(f"变换后特征形状: {X_transformed.shape}")

        # 定义分类器
        clf1 = SVC(kernel='rbf', probability=True, random_state=42)
        clf2 = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
        eclf = VotingClassifier(estimators=[('svm', clf1), ('j48', clf2), ('rf', clf3)], voting='soft')

        # 10折交叉验证
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)  # 每次运行使用不同种子
        f1_scores = []
        for train_idx, val_idx in skf.split(X_transformed, y_train):
            X_train_fold, X_val_fold = X_transformed[train_idx], X_transformed[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            eclf.fit(X_train_fold, y_train_fold)
            y_pred = eclf.predict(X_val_fold)

            # 计算每个类别的F1分数并取平均
            f1_per_class = f1_score(y_val_fold, y_pred, average=None)
            f1_scores.append(np.mean(f1_per_class))

        avg_f1 = np.mean(f1_scores) * 100
        print(f"平均F1分数: {avg_f1:.2f}%")

    except Exception as e:
        print(f"评估错误: {e}")
        avg_f1 = 0.0
    return (avg_f1,)


toolbox.register("evaluate", evalVFGP)
toolbox.register("select", tools.selTournament, tournsize=7)  # 锦标赛选择
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=2, max_=6)  # 与初始化一致
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

if __name__ == "__main__":
    # 参数设置
    population_size = 100
    generations = 50
    cx_prob = 0.8
    mut_prob = 0.19
    elit_prob = 0.01

    # 初始化种群
    random.seed(None)  # 确保每次运行随机性
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # 进化过程
    pop, log, _ = evalGP.eaSimple(
        pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, elitpb=elit_prob, ngen=generations,
        stats=stats, halloffame=hof, verbose=True
    )

    # 获取最佳个体
    best_individual = hof[0]
    best_func = toolbox.compile(expr=best_individual)
    selected_features = get_selected_features(best_individual)

    # 转换训练和测试数据
    X_train_transformed = []
    for sample in x_train:
        constructed_feature = best_func(*[float(sample[i]) for i in range(x_train.shape[1])])
        feature_vector = [float(sample[i]) for i in selected_features] + [float(constructed_feature)]
        X_train_transformed.append(feature_vector)
    X_train_transformed = np.array(X_train_transformed)

    X_test_transformed = []
    for sample in x_test:
        constructed_feature = best_func(*[float(sample[i]) for i in range(x_test.shape[1])])
        feature_vector = [float(sample[i]) for i in selected_features] + [float(constructed_feature)]
        X_test_transformed.append(feature_vector)
    X_test_transformed = np.array(X_test_transformed)

    # 训练最终分类器
    eclf = VotingClassifier(
        estimators=[
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('j48', DecisionTreeClassifier(criterion='entropy', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ],
        voting='soft'
    )
    eclf.fit(X_train_transformed, y_train)
    y_pred = eclf.predict(X_test_transformed)

    # 输出结果
    print("\n=== 最终测试结果 ===")
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1分数 (Macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"最佳个体: {best_individual}")
    print(f"选定特征索引: {selected_features}")

    # 特征分析（可解释性）
    feature_details = []
    for idx in selected_features:
        for f_type, (start, end) in feature_types.items():
            if start <= idx < end:
                feature_details.append({
                    'index': idx,
                    'type': f_type,
                    'relative_index': idx - start
                })
    print("\n=== 选定特征分析 ===")
    for detail in feature_details:
        print(f"特征索引: {detail['index']}, 类型: {detail['type']}, 相对索引: {detail['relative_index']}")