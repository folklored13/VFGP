import numpy as np
import os
import operator
import re
import time
import argparse
from deap import base, creator, tools, gp, algorithms
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report


def Div(left, right):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.divide(float(left), float(right))
            if not np.isfinite(res): return 0.0
            return res
    except ZeroDivisionError: return 0.0

def sin(x): return np.sin(float(x))
def cos(x): return np.cos(float(x))
def If(a, b, c, d): return c if a < b else d

# --- evalVFGP (修改为直接评估，无内部CV) ---
def evalVFGP(individual, x_train_fold, y_train_fold, toolbox): # 传入 toolbox
    try:
        func = toolbox.compile(expr=individual)
        tree_str = str(individual)
        used_feature_indices_str = re.findall(r'F(\d+)', tree_str)
        selected_feature_indices = sorted(list(set(map(int, used_feature_indices_str))))

        if not selected_feature_indices:
            # 理论上GP不应生成无终端的树，但以防万一
            # 检查是否有常数终端 (如果添加了的话)
            has_terminals = any(isinstance(node, gp.Terminal) for node in individual)
            if not has_terminals:
                 # print(f"Warning: Individual has no terminals: {tree_str}")
                 return (0.0,) # 无终端则适应度为0

            # 如果只有常数终端，selected_feature_indices 会为空
            # 此时也应该可以计算构造特征

        # 计算构造特征
        constructed_feature_values = []
        for sample in x_train_fold:
            try:
                feature_values = tuple(sample)
                constructed_val = func(*feature_values)
                constructed_feature_values.append(constructed_val)
            except Exception:
                constructed_feature_values.append(0.0)
        constructed_feature_values = np.array(constructed_feature_values).reshape(-1, 1)
        constructed_feature_values = np.nan_to_num(constructed_feature_values, nan=0.0, posinf=0.0, neginf=0.0)

        # 构建转换后的训练 fold 特征
        if selected_feature_indices:
             try:
                X_train_transformed_fold = np.hstack([x_train_fold[:, selected_feature_indices], constructed_feature_values])
             except (IndexError, ValueError):
                 # print(f"Hstack error during eval. Indices: {selected_feature_indices}")
                 X_train_transformed_fold = constructed_feature_values # Fallback to constructed only
        else:
            # 如果没有 F{i} 终端，只使用构造特征
            X_train_transformed_fold = constructed_feature_values

        # 直接训练和评估 SVM, DT, RF，取最高 F1
        models = {
            'svm': SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale'),
            'j48': DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5), # 保持 J48 风格
            'rf': RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5, n_jobs=-1)
        }
        f1_scores = {}
        best_f1 = -1.0
        # best_classifier_index = -1 # 不需要存储索引，只返回适应度

        for i, (name, model) in enumerate(models.items()):
            try:
                # 处理只有一个特征的情况
                current_X = X_train_transformed_fold
                if current_X.ndim == 1:
                    current_X = current_X.reshape(-1, 1)

                # 检查类别数量是否足够训练
                if len(np.unique(y_train_fold)) < 2:
                     # print(f"Skipping {name} in eval: Not enough classes in training fold.")
                     f1_scores[name] = 0.0
                     continue

                model.fit(current_X, y_train_fold)
                y_pred_train_fold = model.predict(current_X)
                f1_macro = f1_score(y_train_fold, y_pred_train_fold, average='macro', zero_division=0)
                f1_scores[name] = f1_macro
                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    # best_classifier_index = i
            except Exception as e_fit:
                # print(f"Error fitting/predicting {name} in evalVFGP: {e_fit}")
                f1_scores[name] = 0.0

        # 返回适应度元组 (DEAP 要求)
        # print(f"Eval Ind: {tree_str[:50]}... Best F1: {best_f1:.4f}")
        return (best_f1,)

    except Exception as e:
        # print(f"Major error in evalVFGP for individual {individual}: {e}")
        # import traceback
        # traceback.print_exc()
        return (0.0,)


def transform_data(x_original, individual, compiled_func):
    tree_str = str(individual)
    used_feature_indices_str = re.findall(r'F(\d+)', tree_str)
    selected_indices = sorted(list(set(map(int, used_feature_indices_str))))

    constructed = []
    for sample in x_original:
        try:
            constructed.append(compiled_func(*tuple(sample)))
        except Exception:
            constructed.append(0.0)
    constructed = np.array(constructed).reshape(-1, 1)
    constructed = np.nan_to_num(constructed, nan=0.0, posinf=0.0, neginf=0.0)

    if not selected_indices:
        if constructed.shape[1] == 1 and constructed.shape[0] == x_original.shape[0]:
             return constructed # 只返回构造特征
        else:
             # print("Error: No selected features and constructed feature shape mismatch.")
             #返回一个全零数组
             return np.zeros((x_original.shape[0], 1))

    else:
         try:
            selected = x_original[:, selected_indices]
            # 确保 selected 和 constructed 行数一致
            if selected.shape[0] != constructed.shape[0]:
                 # print(f"Error: Row mismatch in transform_data. Selected: {selected.shape[0]}, Constructed: {constructed.shape[0]}")
                 # 尝试只返回 selected 或 constructed，或返回错误
                 return selected # 或者 constructed，或者错误处理
            return np.hstack([selected, constructed])
         except IndexError:
             # print(f"Error hstacking final transform. Indices: {selected_indices}")
             if constructed.shape[1] == 1 and constructed.shape[0] == x_original.shape[0]:
                 return constructed # Fallback to constructed only
             else:
                  return np.zeros((x_original.shape[0], 1))


# --- Main Execution ---
def main(feature_type, n_features, data_dir, num_jobs=5, n_splits=10):
    print(f"\n========== Running GP for Feature Type: {feature_type} ({n_features} features) ==========")
    print(f"Jobs = {num_jobs}, Folds per Job = {n_splits}")

    # --- Load Data ---
    try:
        all_features = np.load(os.path.join(data_dir, f"all_features_{feature_type}.npy"))
        all_labels = np.load(os.path.join(data_dir, f"all_labels_{feature_type}.npy"))
        print(f"Loaded data shapes - Features: {all_features.shape}, Labels: {all_labels.shape}")
        if all_features.shape[1] != n_features:
             print(f"Error: Loaded feature dimension {all_features.shape[1]} does not match expected {n_features}")
             return
    except FileNotFoundError:
        print(f"Error: Data files not found in {data_dir} for feature type {feature_type}")
        return

    # --- GP Setup ---
    pset = gp.PrimitiveSet("MAIN", arity=n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(Div, 2)
    pset.addPrimitive(sin, 1)
    pset.addPrimitive(cos, 1)
    pset.addPrimitive(If, 4)
    pset.renameArguments(**{f"ARG{i}": f"F{i}" for i in range(n_features)})


    # creator.create("FitnessMax", base.Fitness, weights=(1.0,), best_model_index=None)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)


    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    # toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    # toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # --- GP Parameters ---
    population_size = 100
    generations = 50
    cx_prob = 0.80
    mut_prob = 0.19

    mu = population_size
    lambda_ = population_size

    # --- CV and Job Loop ---
    all_job_train_f1s = []
    all_job_test_f1s = []
    all_job_train_times = []
    all_job_test_times = []

    overall_start_time = time.time()

    for job_num in range(num_jobs):
        print(f"\n******************** Starting Job {job_num} *********************")
        job_start_time = time.time()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=job_num + 1) # Use job_num for seed consistency
        fold_train_f1s = []
        fold_test_f1s = []
        fold_train_times = []
        fold_test_times = []

        # Scaler for the current job (fit outside fold loop, transform inside)
        scaler = MinMaxScaler()
        scaler.fit(all_features) # Fit on all data (as in original split) - slight leakage risk, or fit on train_fold inside loop

        for fold_num, (train_index, test_index) in enumerate(skf.split(all_features, all_labels)):
            fold_start_time = time.time()
            print(f"\nNumber of features = {n_features}")
            print(f"Fold: {fold_num}\t Job: {job_num}")
            print("\n| ECJ") # Mimic ECJ header
            # print("| An evolutionary computation system (version 23) ...") # Mimic ECJ header
            print(f"Seed: {job_num + 1}") # Mimic ECJ Seed
            print(f"Job: {job_num}") # Mimic ECJ Job

            x_train_fold, x_test_fold = all_features[train_index], all_features[test_index]
            y_train_fold, y_test_fold = all_labels[train_index], all_labels[test_index]

            # Scale data for the fold
            x_train_fold = scaler.transform(x_train_fold)
            x_test_fold = scaler.transform(x_test_fold)

            # Register evaluation function for this fold
            toolbox.register("evaluate", evalVFGP, x_train_fold=x_train_fold, y_train_fold=y_train_fold, toolbox=toolbox)

            # --- Run GP Evolution ---
            pop = toolbox.population(n=population_size)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0)
            stats.register("avg", np.mean)
            stats.register("max", np.max)

            # Using eaMuPlusLambda for potential elitism effect
            pop, logbook = algorithms.eaMuPlusLambda(
                pop, toolbox, mu=mu, lambda_=lambda_,
                cxpb=cx_prob, mutpb=mut_prob, ngen=generations,
                stats=stats, halloffame=hof, verbose=False # Set verbose=True for gen details
            )

            # Check for early stopping (perfect fitness) - Mimic "Found Ideal Individual"
            if hof and hof[0].fitness.valid and hof[0].fitness.values[0] >= 1.0:
                 print("Found Ideal Individual")


            best_ind_fold = hof[0]
            print(f"Subpop 0 best fitness of run: Fitness: Standardized={best_ind_fold.fitness.values[0]:.17f} Adjusted={best_ind_fold.fitness.values[0]:.17f} Hits=N/A") # Mimic ECJ format

            # --- Train & Select Best Classifier (on Train Fold) ---
            best_func_fold = toolbox.compile(expr=best_ind_fold)
            X_train_transformed_fold = transform_data(x_train_fold, best_ind_fold, best_func_fold)

            models_config = {
                'svm': SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale'),
                'dt': DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5),
                'rf': RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5, n_jobs=-1)
            }
            trained_models_fold = {}
            train_f1_scores_fold = {}
            best_train_f1_fold = -1.0
            best_classifier_name_fold = None
            best_classifier_index_fold = -1

            for i, (name, model) in enumerate(models_config.items()):
                current_X_train = X_train_transformed_fold
                if current_X_train.ndim == 1: current_X_train = current_X_train.reshape(-1, 1)
                if len(np.unique(y_train_fold)) < 2: continue

                model.fit(current_X_train, y_train_fold)
                trained_models_fold[name] = model
                y_train_pred_fold = model.predict(current_X_train)
                f1_train = f1_score(y_train_fold, y_train_pred_fold, average='macro', zero_division=0)
                train_f1_scores_fold[name] = f1_train * 100
                if f1_train > best_train_f1_fold:
                    best_train_f1_fold = f1_train
                    best_classifier_name_fold = name
                    best_classifier_index_fold = i

            print(f"\nTrain: SVM = {train_f1_scores_fold.get('svm', 0.0):.2f}, DT = {train_f1_scores_fold.get('dt', 0.0):.2f}, RF = {train_f1_scores_fold.get('rf', 0.0):.2f}")
            print(f"\nTrain: F1 = {best_train_f1_fold * 100:.2f}")
            print(f"\nTrain: Max Index = {best_classifier_index_fold}")
            fold_train_f1s.append(best_train_f1_fold * 100) # Store as %
            fold_train_time = time.time() - fold_start_time
            fold_train_times.append(fold_train_time * 1000) # Store ms

            # --- Test Selected Classifier (on Test Fold) ---
            test_start_time = time.time()
            X_test_transformed_fold = transform_data(x_test_fold, best_ind_fold, best_func_fold)

            # Evaluate all models on test set (mimicking Test: SVM=..., DT=..., RF=...)
            test_f1_scores_fold = {}
            tester_model = None # The one selected based on train performance
            for i, (name, model) in enumerate(trained_models_fold.items()):
                 current_X_test = X_test_transformed_fold
                 if current_X_test.ndim == 1: current_X_test = current_X_test.reshape(-1, 1)
                 if len(np.unique(y_test_fold)) < 2:
                      test_f1_scores_fold[name] = 0.0
                      continue

                 try:
                     y_test_pred_fold = model.predict(current_X_test)
                     f1_test = f1_score(y_test_fold, y_test_pred_fold, average='macro', zero_division=0)
                     test_f1_scores_fold[name] = f1_test * 100
                     if name == best_classifier_name_fold:
                         tester_model = model # Keep track of the selected one
                 except Exception as e_test:
                      # print(f"Error predicting with {name} on test fold: {e_test}")
                      test_f1_scores_fold[name] = 0.0

            # Report the test F1 of the *selected* model
            final_test_f1_fold = test_f1_scores_fold.get(best_classifier_name_fold, 0.0)
            fold_test_f1s.append(final_test_f1_fold) # Store as %
            fold_test_time = (time.time() - test_start_time) * 1000 # Store ms
            fold_test_times.append(fold_test_time)

            print(f"\nTest: SVM = {test_f1_scores_fold.get('svm', 0.0):.2f}, DT = {test_f1_scores_fold.get('dt', 0.0):.2f}, RF = {test_f1_scores_fold.get('rf', 0.0):.2f}")
            print(f"TREE 0: {str(best_ind_fold)}")
            print(f"\nTest: F1 = {final_test_f1_fold:.2f}")

        # --- Fold Loop Ends ---
        job_avg_train_f1 = np.mean(fold_train_f1s)
        job_std_train_f1 = np.std(fold_train_f1s)
        job_avg_test_f1 = np.mean(fold_test_f1s)
        job_std_test_f1 = np.std(fold_test_f1s)
        job_avg_train_time = np.mean(fold_train_times)
        job_avg_test_time = np.mean(fold_test_times)

        all_job_train_f1s.append(job_avg_train_f1)
        all_job_test_f1s.append(job_avg_test_f1)
        all_job_train_times.append(job_avg_train_time)
        all_job_test_times.append(job_avg_test_time)

        print("\n******************** On each fold in one job *********************")
        print(f"Train Acc Array: [{', '.join([f'{f:.1f}' for f in fold_train_f1s])}]") # Mimic Acc naming, use F1 values
        print(f" Test Acc Array: [{', '.join([f'{f:.1f}' for f in fold_test_f1s])}]")
        print(f"Train time Array: [{', '.join([f'{t:.1f}' for t in fold_train_times])}]")
        print(f" Test time Array: [{', '.join([f'{t:.1f}' for t in fold_test_times])}]")
        print("************************* Average Accuracy of all folds in one job *******************************")
        print(f"Training: {job_avg_train_f1:.2f}\t{job_std_train_f1:.2f}")
        print(f"    Test: {job_avg_test_f1:.2f}\t{job_std_test_f1:.2f}")
        print("*****   GP results finished! ******")

    # --- Job Loop Ends ---
    final_avg_train_f1 = np.mean(all_job_train_f1s)
    # Std across job averages (ECJ might calculate overall std differently)
    final_std_train_f1 = np.std(all_job_train_f1s)
    final_avg_test_f1 = np.mean(all_job_test_f1s)
    final_std_test_f1 = np.std(all_job_test_f1s)
    final_avg_train_time = np.mean(all_job_train_times)
    final_avg_test_time = np.mean(all_job_test_times)

    print("\n******************** On each job *********************")
    print(f"trainAccfinal: [{', '.join([f'{f:.15f}' for f in all_job_train_f1s])}]") # More precision
    print(f"\n testAccfinal: [{', '.join([f'{f:.15f}' for f in all_job_test_f1s])}]")
    print("\n ******************** On all jobs *********************")
    print(f"{final_avg_train_f1:.2f} $\pm$ {final_std_train_f1:.2f}  & {final_avg_test_f1:.2f} $\pm$ {final_std_test_f1:.2f}")
    print(f" Average training time = {final_avg_train_time:.2f}")
    print(f" Average test time = {final_avg_test_time:.2f}*****   GP results finished! ******")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP experiments on different feature sets.')
    parser.add_argument('feature_type', choices=['lbpgray', 'lbprgb', 'wavelet'], help='Type of features to use.')
    args = parser.parse_args()

    feature_dims = {
        'lbpgray': 59,
        'lbprgb': 177,
        'wavelet': 416
    }
    n_features = feature_dims[args.feature_type]
    data_dir = "E:/datasets/PH2-dataset/processed_ph2_separate/" + args.feature_type
    #data_dir = "./processed_ph2_separate/" + args.feature_type # Use relative path

    main(feature_type=args.feature_type,
         n_features=n_features,
         data_dir=data_dir,
         num_jobs=5,
         n_splits=10)