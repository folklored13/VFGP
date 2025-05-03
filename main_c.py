import random
import time
import re
import numpy as np
import os
import operator
from deap import base, creator, tools, gp
# from deap.gp import PrimitiveSetTyped # 改用标准 PrimitiveSet
import evalGP_fgp as evalGP
# import fgp_functions as fe_fs # 特征提取已在split中完成
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score


def protectedDiv(left, right):

    try:
        # 使用 numpy 的除法，并处理无穷大和 NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.divide(float(left), float(right))
            if not np.isfinite(res):
                return 0.0
            return res
    except ZeroDivisionError:
        return 0.0 #  1.0

# 加载数据
data_path = "E:/datasets/PH2-dataset/processed_ph2/"
x_train = np.load(os.path.join(data_path, "train_features.npy"))
y_train = np.load(os.path.join(data_path, "train_labels.npy"))
x_test = np.load(os.path.join(data_path, "test_features.npy"))
y_test = np.load(os.path.join(data_path, "test_labels.npy"))

n_features = x_train.shape[1]
if n_features != 652:
    print(f"警告：加载的特征维度 {n_features} 与预期的 652 不符！")

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 定义GP树 - 使用标准 PrimitiveSet
# 输入是特征索引的数量，输出是一个浮点数（构造特征值）
pset = gp.PrimitiveSet("MAIN", arity=n_features) # arity是终端数量

# 添加基本算术运算符
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)

# 添加三角函数
def sin(x): return np.sin(float(x))
def cos(x): return np.cos(float(x))
pset.addPrimitive(sin, 1)
pset.addPrimitive(cos, 1)

# 添加条件操作符 (IF a < b THEN c ELSE d)
def If(a, b, c, d):
    return c if a < b else d
pset.addPrimitive(If, 4)

# 重命名参数为 F0, F1, ..., F651
pset.renameArguments(**{f"ARG{i}": f"F{i}" for i in range(n_features)})

# 定义个体和适应度
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6) # 树深度与论文一致
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# compile 用于将树转换为可调用函数
toolbox.register("compile", gp.compile, pset=pset)


def evalVFGP(individual, x_data, y_data):
    try:
        # 1
        func = toolbox.compile(expr=individual)
        print(f'individual:{individual}')
        # 2. 获取个体实际使用的索引 (通过解析字符串)
        tree_str = str(individual)
        # 使用正则表达式查找所有出现的 "Fi"，提取出数字 i
        # (\d+) 匹配一个或多个数字，并捕获它
        used_feature_indices_str = re.findall(r'F(\d+)', tree_str)
        # 转换为整数并去重
        selected_feature_indices = sorted(list(set(map(int, used_feature_indices_str))))

        #
        # print(f"Tree: {tree_str}")
        # print(f"Used Indices: {selected_feature_indices}")


        if not selected_feature_indices:
            # print("警告: 个体未使用任何特征 F{i}")
            return (0.0,)

        # 3. 计算构造特征
        constructed_feature_values = []
        for sample in x_data:
            try:
                feature_values = tuple(sample) # 传递样本的所有特征值
                constructed_val = func(*feature_values)
                constructed_feature_values.append(constructed_val)
            except Exception as e_func: # 捕获函数执行错误
                # print(f"错误: 执行 func 时出错: {e_func}. Sample: {sample[:5]}... Individual: {tree_str}")
                constructed_feature_values.append(0.0) # 出错时使用默认值0
        constructed_feature_values = np.array(constructed_feature_values).reshape(-1, 1)

        # 处理 NaN 或 Inf
        if not np.all(np.isfinite(constructed_feature_values)):
            # print("警告: 构造特征包含 NaN 或 Inf，将被替换为 0")
            constructed_feature_values = np.nan_to_num(constructed_feature_values, nan=0.0, posinf=0.0, neginf=0.0)

        #
        # unique_constructed = np.unique(constructed_feature_values)
        # if len(unique_constructed) < 5: # 如果构造特征的值很少
        #     print(f"构造特征值 (少数 unique): {unique_constructed}")


        # 4. 构建新的特征矩阵 (m 个选择特征 + 1 个构造特征)
        try:
            X_transformed = np.hstack([x_data[:, selected_feature_indices], constructed_feature_values])
        except IndexError as e_index:
            print(f"错误: Hstack 时索引错误. Selected indices: {selected_feature_indices}, Max index needed: {max(selected_feature_indices) if selected_feature_indices else 'N/A'}. x_data shape: {x_data.shape}")
            return (0.0,)
        except ValueError as e_val:
             print(f"错误: Hstack 时值错误. Selected shape: {x_data[:, selected_feature_indices].shape}, Constructed shape: {constructed_feature_values.shape}")
             return (0.0,)


        # 5. 使用10折交叉验证评估集成模型
        f1_scores_models = {'svm': [], 'j48': [], 'rf': []}
        models = {
            'svm': SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale'),
            'j48': DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5),
            'rf': RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5, n_jobs=-1)                                                             # 使用 n_jobs 加速 RF
        }

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_count = 0
        positive_f1_fold_found = False
        for train_idx, val_idx in skf.split(X_transformed, y_data):
            fold_count += 1
            X_train_fold, X_val_fold = X_transformed[train_idx], X_transformed[val_idx]
            y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]

            for name, model in models.items():
                try:
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    f1_macro = f1_score(y_val_fold, y_pred, average='macro', zero_division=0)
                    f1_scores_models[name].append(f1_macro)
                    if f1_macro > 0:
                        positive_f1_fold_found = True
                except Exception as e_fit:
                    # print(f"警告: 模型 {name} 在 fold {fold_count} 训练/预测失败: {e_fit}")
                    f1_scores_models[name].append(0.0)

        # 计算每个模型的平均F1
        avg_f1_svm = np.mean(f1_scores_models['svm']) if f1_scores_models['svm'] else 0.0
        avg_f1_j48 = np.mean(f1_scores_models['j48']) if f1_scores_models['j48'] else 0.0
        avg_f1_rf = np.mean(f1_scores_models['rf']) if f1_scores_models['rf'] else 0.0

        # 个体的适应度是这三个模型里面最高的 F1 分数
        best_avg_f1 = max(avg_f1_svm, avg_f1_j48, avg_f1_rf)

        #
        # if not positive_f1_fold_found and best_avg_f1 == 0:
        #     print(f"警告: 个体 {tree_str[:80]}... 所有 CV folds F1 均为 0.")
        # elif best_avg_f1 > 0:
        #      print(f"个体 {tree_str[:80]}... Fitness: {best_avg_f1:.4f}")
        # ---------------

        return (best_avg_f1,)

    except gp. ävenError as e_compile:
        print(f"错误: GP 编译错误: {e_compile}. Individual: {individual}")
        return (0.0,)
    except MemoryError as e_mem:
        print(f"错误: 内存不足. Individual: {individual}")
        # 可能太大或计算太复杂
        return (0.0,) # 返回0适应度
    except Exception as e:
        print(f"严重错误 in evalVFGP: {e}")
        import traceback
        traceback.print_exc()
        return (0.0,)


toolbox.register("evaluate", evalVFGP, x_data=x_train, y_data=y_train)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


# toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def transform_data(x_original, individual, compiled_func):
    """转换数据：提取选择的特征并计算构造特征"""
    tree_str = str(individual)
    used_feature_indices_str = re.findall(r'F(\d+)', tree_str)
    selected_indices = sorted(list(set(map(int, used_feature_indices_str))))

    constructed = []
    for sample in x_original:
        try:
            constructed.append(compiled_func(*tuple(sample)))
        except Exception:
             constructed.append(0.0) # 出错用0
    constructed = np.array(constructed).reshape(-1, 1)
    constructed = np.nan_to_num(constructed, nan=0.0, posinf=0.0, neginf=0.0)

    if not selected_indices:
        # print("警告: 最终最佳个体未使用任何特征，只使用构造特征进行转换。")
        return constructed
    else:
         try:
            selected = x_original[:, selected_indices]
            return np.hstack([selected, constructed])
         except IndexError:
             print(f"错误: 最终转换时索引错误. Selected indices: {selected_indices}. x_original shape: {x_original.shape}")
             # 返回一个默认形状要不就引发错误
             return constructed # 只返回构造特征作为后备

if __name__ == "__main__":
    start_time = time.time()

    population_size = 100
    generations = 50
    cx_prob = 0.80
    mut_prob = 0.19
    elit_prob = 0.01

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0) # 处理无效的
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log, _ = evalGP.eaSimple(
        pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, elitpb=elit_prob, ngen=generations,
        stats=stats, halloffame=hof, verbose=True
    )

    best_ind = hof[0]
    print("\n=== 演化结束 ===")
    print("最佳个体:", best_ind)
    # 检查适应度是否有效
    best_fitness = best_ind.fitness.values[0] if best_ind.fitness.valid else 0.0
    print(f"最佳个体适应度 (训练集 Macro F1): {best_fitness:.4f}")

    # 1. 编译最佳个体
    best_func = toolbox.compile(expr=best_ind)

    # 2. 转换训练集和测试集 (用更新后的 transform_data)
    X_train_transformed = transform_data(x_train, best_ind, best_func)
    X_test_transformed = transform_data(x_test, best_ind, best_func)

    # 打印检查转换后的维度
    tree_str_final = str(best_ind)
    used_indices_final_str = re.findall(r'F(\d+)', tree_str_final)
    selected_indices_final = sorted(list(set(map(int, used_indices_final_str))))
    print(f"最终最佳个体使用的特征索引: {selected_indices_final}")
    print(f"最终选择的特征数量: {len(selected_indices_final)}")
    print(f"最终转换后训练集维度: {X_train_transformed.shape}")  #  (160, len(selected_indices_final) + 1)
    print(f"最终转换后测试集维度: {X_test_transformed.shape}")  #  (40, len(selected_indices_final) + 1)

    # 4. 训练最终的模型

    final_models = {
        'svm': SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale'),
        'j48': DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5),
        'rf': RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    }
    best_model_name = None
    best_train_f1 = -1.0

    print("\n在转换后的训练集上训练最终模型...")
    for name, model in final_models.items():

         model.fit(X_train_transformed, y_train)
         y_train_pred = model.predict(X_train_transformed)
         train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
         print(f"模型 {name} - 训练集 Macro F1: {train_f1:.4f}")
         if train_f1 > best_train_f1:
             best_train_f1 = train_f1
             best_model_name = name

    print(f"\n选择最佳模型 '{best_model_name}' 进行测试。")
    final_classifier = final_models[best_model_name]

    # 5. 在转换后的测试集上评估
    y_test_pred = final_classifier.predict(X_test_transformed)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)


    print("\n=== 最终测试结果 ===")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集 Macro F1 分数: {test_f1_macro:.4f}")


    from sklearn.metrics import classification_report
    print("\n测试集分类报告:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"总运行时间：{total_time:.2f}s")