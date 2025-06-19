import numpy as np
import re
from deap import gp
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import config # 导入配置

def evalVFGP(individual, x_train_fold, y_train_fold, toolbox):
    """
    评估单个 GP 个体的适应度 (在当前训练折上)。
    直接评估 SVM, DT, RF，返回最高的 Macro F1 分数。
    """
    try:
        func = toolbox.compile(expr=individual)
        tree_str = str(individual)
        used_feature_indices_str = re.findall(r'F(\d+)', tree_str)
        selected_feature_indices = sorted(list(set(map(int, used_feature_indices_str))))

        # 检查是否有终端（包括常数）
        has_terminals = any(isinstance(node, gp.Terminal) for node in individual)
        if not has_terminals:
            # print(f"Warning: Eval Individual has no terminals: {tree_str}")
            return (0.0,)

        # 计算构造特征
        constructed_feature_values = []
        for sample in x_train_fold:
            try:
                feature_values = tuple(sample)
                constructed_val = func(*feature_values)
                # 再次检查 NaN/Inf (func 可能产生)
                if not np.isfinite(constructed_val):
                    constructed_val = 0.0
                constructed_feature_values.append(constructed_val)
            except (ValueError, OverflowError, TypeError, RuntimeWarning): # 捕获潜在计算错误
                constructed_feature_values.append(0.0)
        constructed_feature_values = np.array(constructed_feature_values).reshape(-1, 1)
        # 最后再检查一次
        constructed_feature_values = np.nan_to_num(constructed_feature_values, nan=0.0, posinf=0.0, neginf=0.0)


        # 构建转换后的训练 fold 特征
        if selected_feature_indices:
             try:
                 # # 索引在 x_train_fold 的范围内
                 # if selected_feature_indices and max(selected_feature_indices) >= x_train_fold.shape[1]:
                 #     # print(f"Warning: Index out of bounds. Max index: {max(selected_feature_indices)}, Features: {x_train_fold.shape[1]}")
                 #     # 可以只用构造特征或返回0
                 #     X_train_transformed_fold = constructed_feature_values
                 # else:
                 #    selected_features = x_train_fold[:, selected_feature_indices]
                 #    X_train_transformed_fold = np.hstack([selected_features, constructed_feature_values])
                 X_train_transformed_fold = transform_data(x_train_fold, individual, func)
             except (IndexError, ValueError) as e_hstack:
                 # print(f"Hstack error during eval. Indices: {selected_feature_indices}. Error: {e_hstack}")
                 # 如果hstack 出错，就只用构造特征
                 X_train_transformed_fold = constructed_feature_values
        else:
            # 如果没有 F{i} 终端，只使用构造特征
            X_train_transformed_fold = constructed_feature_values

        # 确保最终特征矩阵是 2D
        if X_train_transformed_fold.ndim == 1:
            X_train_transformed_fold = X_train_transformed_fold.reshape(-1, 1)
        if X_train_transformed_fold.shape[0] != y_train_fold.shape[0]:
            # print(f"Shape mismatch after transform: X={X_train_transformed_fold.shape}, y={y_train_fold.shape}")
            return (0.0,) # 无法训练

        # 直接训练和评估 SVM, DT, RF，取最高 F1
        models = {
            name: cfg['class'](**cfg['params'])
            for name, cfg in config.CLASSIFIER_CONFIG.items()
        }

        best_f1 = -1.0

        # 检查类别数量是否足够训练
        if len(np.unique(y_train_fold)) < 2:
            # print("Warning: Not enough classes in training fold for evaluation.")
            return (0.0,)

        for name, model in models.items():
            try:
                model.fit(X_train_transformed_fold, y_train_fold)
                y_pred_train_fold = model.predict(X_train_transformed_fold)
                # Macro F1 是 ECJ 日志中最可能使用的指标
                f1_macro = f1_score(y_train_fold, y_pred_train_fold, average='macro', zero_division=0)
                if f1_macro > best_f1:
                    best_f1 = f1_macro
            except ValueError as e_fit: # 捕获类似 "Number of classes..must be greater than 1." 的错误
                 # print(f"Fit error ({name}): {e_fit}. Skipping.")
                 pass # 保持 best_f1 不变
            except Exception as e_other: # 捕获其他潜在错误
                # print(f"Unexpected error fitting/predicting {name} in evalVFGP: {e_other}")
                pass # 保持 best_f1 不变


        # 返回适应度元组 (DEAP 要求)
        return (best_f1 if best_f1 >= 0 else 0.0,) # 确保至少返回 0.0

    except Exception as e:
        # print(f"Major error in evalVFGP for individual {individual}: {e}")
        # import traceback
        # traceback.print_exc()
        return (0.0,)


def transform_data(x_original, individual, compiled_func):
    """
    使用给定的 GP 个体转换数据集。
    """
    tree_str = str(individual)
    used_feature_indices_str = re.findall(r'F(\d+)', tree_str)
    selected_indices = sorted(list(set(map(int, used_feature_indices_str))))

    # 计算构造特征
    constructed = []
    for sample in x_original:
        try:
            # 确保传递了正确数量的参数
            constructed_val = compiled_func(*tuple(sample))
            if not np.isfinite(constructed_val): constructed_val = 0.0
            constructed.append(constructed_val)
        except (ValueError, OverflowError, TypeError, RuntimeWarning):
            constructed.append(0.0)
        except Exception as e_func: # 捕获其他编译函数执行错误
            # print(f"Error during transform_data func execution: {e_func}")
            constructed.append(0.0)
    constructed = np.array(constructed).reshape(-1, 1)
    constructed = np.nan_to_num(constructed, nan=0.0, posinf=0.0, neginf=0.0)

    # 组合特征
    if not selected_indices:
        if constructed.shape[0] == x_original.shape[0]:
            return constructed # 只返回构造特征
        else:
            # print("Transform Error: No selected features & shape mismatch.")
            return np.zeros((x_original.shape[0], 1)) # 返回占位符
    else:
         try:
            # 再次检查索引范围
            if max(selected_indices) >= x_original.shape[1]:
                 # print(f"Transform Error: Max index {max(selected_indices)} >= features {x_original.shape[1]}")
                 return constructed # Fallback
            selected = x_original[:, selected_indices]
            if selected.shape[0] != constructed.shape[0]:
                 # print(f"Transform Error: Row mismatch. Selected={selected.shape[0]}, Constructed={constructed.shape[0]}")
                 return selected # 或者 constructed
            return np.hstack([selected, constructed])
         except IndexError as e_index:
             # print(f"Transform IndexError: {e_index}. Indices: {selected_indices}, Shape: {x_original.shape}")
             return constructed # Fallback
         except ValueError as e_value:
             # print(f"Transform ValueError: {e_value}. Selected shape: {x_original[:, selected_indices].shape}, Constructed shape: {constructed.shape}")
             return constructed # Fallback