import random
import numpy as np
import os
from deap import base, creator, tools, gp
from deap.gp import PrimitiveSetTyped

import evalGP_fgp as evalGP
import fgp_functions as fe_fs
from fgp_functions import vec_add, vec_sub, vec_mul, vec_div, vec_sin, vec_cos, vec_conditional
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# 定义类型
ImgRGB = np.ndarray  # RGB图像类型 (H,W,3)
ImgLum = np.ndarray  # 带亮度的图像类型 (H,W,4)
FeatureVector = list  # 特征向量类型

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

#初始化PrimitiveSet
pset = gp.PrimitiveSetTyped("MAIN", [FeatureVector], FeatureVector)
pset.renameArguments(ARG0='combined_features')

# 添加终端允许选择特定特征
for i in range(588):  # 总特征数59+177+
    pset.addTerminal(i, int, name=f"F{i}")  #特征索引作为终端

# 添加特征构造
def select_feature(vec, idx):
    return [vec[idx]]  # 选择单个特征
pset.addPrimitive(select_feature, [FeatureVector, int], FeatureVector, name="Select")

pset.addPrimitive(fe_fs.uniform_lbp_gray, [ImgRGB], FeatureVector, name="LGray")
pset.addPrimitive(fe_fs.uniform_lbp_rgb, [ImgRGB], FeatureVector, name="LRGB")
pset.addPrimitive(fe_fs.wavelet_transform, [ImgLum], FeatureVector, name="Wavelet")

pset.addPrimitive(vec_add, [FeatureVector, FeatureVector], FeatureVector, name="vec_add")
pset.addPrimitive(vec_sub, [FeatureVector, FeatureVector], FeatureVector, name="vec_sub")
pset.addPrimitive(vec_mul, [FeatureVector, FeatureVector], FeatureVector, name="vec_mul")
pset.addPrimitive(vec_div, [FeatureVector, FeatureVector], FeatureVector, name="vec_div")
pset.addPrimitive(vec_sin, [FeatureVector], FeatureVector, name="vec_sin")
pset.addPrimitive(vec_cos, [FeatureVector], FeatureVector, name="vec_cos")
pset.addPrimitive(vec_conditional, [FeatureVector, FeatureVector, FeatureVector, FeatureVector], FeatureVector, name="vec_if")

# #添加运算符
# pset.addPrimitive(np.add, [Vector, Vector], Vector, name="add")
# pset.addPrimitive(np.subtract, [Vector, Vector], Vector, name="subtract")
# pset.addPrimitive(np.multiply, [Vector, Vector], Vector, name="multiply")
# pset.addPrimitive(lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b != 0),
#                   [Vector, Vector], Vector, name="div")
# pset.addPrimitive(np.sin, [Vector], Vector, name="sin")
# pset.addPrimitive(np.cos, [Vector], Vector, name="cos")
# pset.addPrimitive(fe_fs.conditional_op, [Vector, Vector, Vector, Vector], Vector, name="if")

# #添加随机常数
# pset.addEphemeralConstant("rand_const", lambda: [random.random()], Vector)

#注册工具箱
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


#评估函数
def evalVFGP(individual):
    try:
        func = toolbox.compile(expr=individual)
        X, y = [], []

        for idx in range(len(x_train)):
            combined_features = x_train[idx]
            # img_rgb = x_train_rgb[idx]  #用于提取LGray和LRGB特征
            # img_lum = x_train_lum[idx]  #用于提取小波特征

            # # 提取三个基础特征集
            # lgray = fe_fs.uniform_lbp_gray(img_rgb)  # 59维
            # lrgb = fe_fs.uniform_lbp_rgb(img_rgb)  # 177维
            # wavelet = fe_fs.wavelet_transform(img_lum)  # 416维
            #
            # # 合并为总特征向量（652维）
            # combined_features = np.concatenate([lgray, lrgb, wavelet])

            # GP树处理合并后的特征（选择+构造）
            processed_features = func(combined_features)
            X.append(processed_features)
            y.append(y_train[idx])

        X = MinMaxScaler().fit_transform(X) #归一化

        #定义分类器
        clf1 = SVC(kernel='rbf', probability=True, random_state=42)
        clf2 = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
        eclf = VotingClassifier(
            estimators=[('svm', clf1), ('j48', clf2), ('rf', clf3)],
            voting='soft'
        )

        f1_scores = cross_val_score(eclf, X, y, cv=10, scoring='f1_macro')
        avg_f1 = np.mean(f1_scores) * 100
    except Exception as e:
        print(f"Error during evaluation: {e}")
        avg_f1 = 0
    return (avg_f1,)


toolbox.register("evaluate", evalVFGP)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

if __name__ == "__main__":
    #加载数据
    data_path = "E:/pycode/i_c/data/"

    # x_train_rgb = np.load(os.path.join(data_path, "HAM10000_train_rgb.npy"))#N,H,W,3
    # x_train_lum = np.load(os.path.join(data_path, "HAM10000_train_lum.npy"))#N,H,W,4
    x_train = np.load(os.path.join(data_path, "HAM10000_train_features.npy"))
    y_train = np.load(os.path.join(data_path, "HAM10000_train_labels.npy"))
    print (x_train,y_train)
    # x_test_rgb = np.load(os.path.join(data_path, "HAM10000_test_rgb.npy"))
    # x_test_lum = np.load(os.path.join(data_path, "HAM10000_test_lum.npy"))
    x_test = np.load(os.path.join(data_path, "HAM10000_test_features.npy"))
    y_test = np.load(os.path.join(data_path, "HAM10000_test_labels.npy"))

    # #测试LGray特征
    # sample_gray_feat = fe_fs.uniform_lbp_gray(x_train_rgb[0])
    # print("LGray特征维度:", len(sample_gray_feat))  #59
    #
    # #LRGB特征
    # sample_rgb_feat = fe_fs.uniform_lbp_rgb(x_train_rgb[0])
    # print("LRGB特征维度:", len(sample_rgb_feat))  #177(59*3)
    #
    # #测试小波特征
    # sample_wavelet_feat = fe_fs.wavelet_transform(x_train_lum[0])
    # print("Wavelet特征维度:", len(sample_wavelet_feat))  #416

    #参数设置
    # gp_params = {
    #     'population_size': 100,
    #     'generations': 50,
    #     'crossover_rate': 0.8,
    #     'mutation_rate': 0.19,
    #     'max_tree_depth': 6,
    #     'function_set': ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'if'],
    #     'metric': 'f1_macro'  #
    # }
    #运行GP
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = evalGP.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                               stats=stats, halloffame=hof, verbose=True)

    #用提取到的特征向量生成新的特征向量(先合并三个特征集合成一个特征矩阵？然后再输入到GP中进行特征选择、构建)

    #新特征向量
    #train_features = gp.transform_features(X_train)
    #test_features = gp.transform_features(X_test)

    #训练最终模型并评估
    best_func = toolbox.compile(expr=hof[0])
    X_train_scaled = []
    for idx in range(len(x_train)):
        # img_rgb = x_train_rgb[idx]
        # img_lum = x_train_lum[idx]
        features = best_func(x_train[idx])  #传预处理之后的
        X_train_scaled.append(features)
    X_train_scaled = MinMaxScaler().fit_transform(X_train_scaled)

    eclf = VotingClassifier(
        estimators=[
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('j48', DecisionTreeClassifier(criterion='entropy', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ],
        voting='soft'
    )
    eclf.fit(X_train_scaled, y_train)

    #
    X_test_scaled = []
    for idx in range(len(x_test)):
        # img_rgb = x_test_rgb[idx]
        # img_lum = x_test_lum[idx]
        features = best_func(x_test[idx])
        X_test_scaled.append(features)
    X_test_scaled = MinMaxScaler().fit_transform(X_test_scaled)
    y_pred = eclf.predict(X_test_scaled)

    print("\n=== 最终测试结果 ===")
    print("准确率:", accuracy_score(y_test, y_pred))
    print("F1分数 (Macro):", f1_score(y_test, y_pred, average='macro'))

    print("示例gptree:", str(hof[0]))  # 检查是否包含+-*/等操作符