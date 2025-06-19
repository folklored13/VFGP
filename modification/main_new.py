import argparse
import os
import time
import numpy as np
from deap import algorithms, tools, base, creator, gp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score


import config
from gp_setup import create_toolbox
from evaluation import evalVFGP, transform_data

def run_ecj_like_experiment_from_runs(feature_type, n_features, base_runs_dir, num_total_jobs, num_runs_per_job):
    """
    模拟 ECJ 日志的实验流程，从预先生成的 runs/folds 加载数据。
    num_total_jobs: 对应 ECJ 的 'jobs'，即重复整个 'num_runs_per_job' 次的过程。
    num_runs_per_job: 对应 ECJ 的 'folds'，即每次 Job 内运行的预划分数据集的数量。
    """
    print(f"\n========== Running GP (ECJ Style from Runs) for Feature Type: {feature_type} ({n_features} features) ==========")
    print(f"Total Jobs = {num_total_jobs}, Runs per Job = {num_runs_per_job}")

    overall_job_avg_test_f1s = [] # 存储每个 job 的平均测试 F1
    overall_job_avg_train_f1s = [] # 存储每个 job 的平均训练 F1
    overall_job_train_f1s = []
    overall_job_test_f1s = []
    overall_job_train_times = []
    overall_job_test_times = []

    for job_idx in range(num_total_jobs):
        print(f"\n******************** Starting Job {job_idx} *********************")
        # ECJ seed 是针对每个 Job 的
        current_job_seed = job_idx + 1 # ECJ seed 从 1 开始

        job_fold_train_f1s = []  # 存储当前 job 内每个 run/fold 的训练 F1
        job_fold_test_f1s = []   # 存储当前 job 内每个 run/fold 的测试 F1
        job_fold_train_times = []
        job_fold_test_times = []

        feature_type_dir = os.path.join(base_runs_dir, feature_type)

        for run_idx in range(num_runs_per_job): #对应日志中的 "Fold"
            fold_start_time = time.time()
            print(f"\nNumber of features = {n_features}")
            print(f"Fold: {run_idx}\t Job: {job_idx}")

            print(f"Threads:  breed/1 eval/1")
            print(f"Seed: {current_job_seed}") # ECJ seed 应用于整个 job/ECJ 运行
            print(f"Job: {job_idx}")
            print("Setting up")

            run_dir = os.path.join(feature_type_dir, f"run{run_idx}")
            try:
                x_train = np.load(os.path.join(run_dir, f"train_features_{feature_type}.npy"))
                y_train = np.load(os.path.join(run_dir, f"train_labels_{feature_type}.npy"))
                x_test = np.load(os.path.join(run_dir, f"test_features_{feature_type}.npy"))
                y_test = np.load(os.path.join(run_dir, f"test_labels_{feature_type}.npy"))
            except FileNotFoundError:
                print(f"Error: Run {run_idx} data not found in {run_dir}. Skipping.")
                continue
            except Exception as e:
                print(f"Error loading run {run_idx} data: {e}. Skipping.")
                continue

            # Scale Data
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Create Toolbox
            toolbox = create_toolbox(n_features)
            toolbox.register("evaluate", evalVFGP, x_train_fold=x_train, y_train_fold=y_train, toolbox=toolbox)

            # Run GP Evolution
            pop = toolbox.population(n=config.POPULATION_SIZE)
            hof = tools.HallOfFame(1)
            # 为当前 GP 运行设置随机种子
            # random.seed(current_job_seed * 100 + run_idx) # 示例种子
            # np.random.seed(current_job_seed * 100 + run_idx)


            gp_start_time = time.time()
            pop, logbook = algorithms.eaMuPlusLambda(
                pop, toolbox, mu=config.MU, lambda_=config.LAMBDA_,
                cxpb=config.CX_PROB, mutpb=config.MUT_PROB, ngen=config.GENERATIONS,
                stats=None, halloffame=hof, verbose=False
            )
            gp_train_time_ms = (time.time() - gp_start_time) * 1000
            job_fold_train_times.append(gp_train_time_ms)


            best_ind_run = hof[0]
            if not best_ind_run.fitness.valid: best_fitness_val = 0.0
            else: best_fitness_val = best_ind_run.fitness.values[0]

            if best_fitness_val >= 1.0: print("Found Ideal Individual") # Fitness 0-1
            print(f"Subpop 0 best fitness of run: Fitness: Standardized={best_fitness_val:.17f} Adjusted={best_fitness_val:.17f} Hits=N/A")


            # Train & Select Best Classifier (on Train set of this run)
            try: best_func_run = toolbox.compile(expr=best_ind_run)
            except Exception: print("Error compiling best ind for run."); continue
            X_train_transformed_run = transform_data(x_train, best_ind_run, best_func_run)
            if X_train_transformed_run.ndim == 1: X_train_transformed_run = X_train_transformed_run.reshape(-1, 1)

            train_f1_scores_run = {}
            best_train_f1_run = -1.0
            best_classifier_name_run = list(config.CLASSIFIER_CONFIG.keys())[0]
            best_classifier_index_run = 0
            trained_models_run = {}

            for i, (name, cfg) in enumerate(config.CLASSIFIER_CONFIG.items()):
                model = cfg['class'](**cfg['params'])
                try:
                    if len(np.unique(y_train)) < 2: continue
                    model.fit(X_train_transformed_run, y_train)
                    trained_models_run[name] = model
                    y_train_pred_run = model.predict(X_train_transformed_run)
                    f1_train = f1_score(y_train, y_train_pred_run, average='macro', zero_division=0)
                    train_f1_scores_run[name] = f1_train * 100
                    if f1_train > best_train_f1_run:
                        best_train_f1_run = f1_train
                        best_classifier_name_run = name
                        best_classifier_index_run = i
                except Exception: train_f1_scores_run[name] = 0.0
            if best_train_f1_run < 0: best_train_f1_run = 0.0
            job_fold_train_f1s.append(best_train_f1_run * 100)

            print(f"\nTrain: SVM = {train_f1_scores_run.get('svm', 0.0):.2f}, DT = {train_f1_scores_run.get('dt', 0.0):.2f}, RF = {train_f1_scores_run.get('rf', 0.0):.2f}")
            print(f"\nTrain: F1 = {best_train_f1_run * 100:.2f}")
            print(f"\nTrain: Max Index = {best_classifier_index_run}")

            # Test ALL Classifiers on Test set, Report F1 of SELECTED
            test_start_time = time.time()
            X_test_transformed_run = transform_data(x_test, best_ind_run, best_func_run)
            if X_test_transformed_run.ndim == 1: X_test_transformed_run = X_test_transformed_run.reshape(-1, 1)

            test_f1_scores_run = {}
            final_test_f1_run = 0.0

            for name, model in trained_models_run.items():
                 try:
                     if len(np.unique(y_test)) < 2: test_f1_scores_run[name] = 0.0; continue
                     y_test_pred_run = model.predict(X_test_transformed_run)
                     f1_test = f1_score(y_test, y_test_pred_run, average='macro', zero_division=0)
                     test_f1_scores_run[name] = f1_test * 100
                     if name == best_classifier_name_run:
                         final_test_f1_run = test_f1_scores_run[name]
                 except Exception: test_f1_scores_run[name] = 0.0

            test_time_ms = (time.time() - test_start_time) * 1000
            job_fold_test_times.append(test_time_ms)
            job_fold_test_f1s.append(final_test_f1_run)

            print(f"\nTest: SVM = {test_f1_scores_run.get('svm', 0.0):.2f}, DT = {test_f1_scores_run.get('dt', 0.0):.2f}, RF = {test_f1_scores_run.get('rf', 0.0):.2f}")
            print(f"TREE 0: {str(best_ind_run)}")
            print(f"\nTest: F1 = {final_test_f1_run:.2f}")

        # --- Run/Fold Loop Ends ---
        # Job Summary
        job_avg_train_f1 = np.mean(job_fold_train_f1s) if job_fold_train_f1s else 0
        job_std_train_f1 = np.std(job_fold_train_f1s) if job_fold_train_f1s else 0
        job_avg_test_f1 = np.mean(job_fold_test_f1s) if job_fold_test_f1s else 0
        job_std_test_f1 = np.std(job_fold_test_f1s) if job_fold_test_f1s else 0
        job_avg_train_time = np.mean(job_fold_train_times) if job_fold_train_times else 0
        job_avg_test_time = np.mean(job_fold_test_times) if job_fold_test_times else 0

        overall_job_train_f1s.append(job_avg_train_f1)
        overall_job_test_f1s.append(job_avg_test_f1)
        overall_job_train_times.append(job_avg_train_time)
        overall_job_test_times.append(job_avg_test_time)

        print("\n******************** On each fold in one job *********************")
        print(f"Train Acc Array: [{', '.join([f'{f:.1f}' for f in job_fold_train_f1s])}]")
        print(f" Test Acc Array: [{', '.join([f'{f:.1f}' for f in job_fold_test_f1s])}]")
        print(f"Train time Array: [{', '.join([f'{t:.1f}' for t in job_fold_train_times])}]")
        print(f" Test time Array: [{', '.join([f'{t:.1f}' for t in job_fold_test_times])}]")
        print("************************* Average Accuracy of all folds in one job *******************************")
        print(f"Training: {job_avg_train_f1:.2f}\t{job_std_train_f1:.2f}")
        print(f"    Test: {job_avg_test_f1:.2f}\t{job_std_test_f1:.2f}")
        print("*****   GP results finished! ******")

    # --- Job Loop Ends ---
    final_avg_train_f1 = np.mean(overall_job_train_f1s) if overall_job_train_f1s else 0
    final_std_train_f1 = np.std(overall_job_train_f1s) if overall_job_train_f1s else 0
    final_avg_test_f1 = np.mean(overall_job_test_f1s) if overall_job_test_f1s else 0
    final_std_test_f1 = np.std(overall_job_test_f1s) if overall_job_test_f1s else 0
    final_avg_train_time_overall = np.mean(overall_job_train_times) if overall_job_train_times else 0
    final_avg_test_time_overall = np.mean(overall_job_test_times) if overall_job_test_times else 0


    print("\n******************** On each job *********************")
    print(f"trainAccfinal: [{', '.join([f'{f:.15f}' for f in overall_job_train_f1s])}]")
    print(f"\n testAccfinal: [{', '.join([f'{f:.15f}' for f in overall_job_test_f1s])}]")
    print("\n ******************** On all jobs *********************")
    print(f"{final_avg_train_f1:.2f} $\pm$ {final_std_train_f1:.2f}  & {final_avg_test_f1:.2f} $\pm$ {final_std_test_f1:.2f}")
    print(f" Average training time = {final_avg_train_time_overall:.2f}")
    print(f" Average test time = {final_avg_test_time_overall:.2f}*****   GP results finished! ******")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP experiments based on pre-split runs/folds.')
    parser.add_argument('feature_type', choices=['lbpgray', 'lbprgb', 'wavelet'], help='Type of features.')
    args = parser.parse_args()

    if args.feature_type not in config.FEATURE_DIMS:
        print(f"Error: Unknown feature type '{args.feature_type}'."); exit()

    n_features = config.FEATURE_DIMS[args.feature_type]
    base_runs_dir = config.PROCESSED_RUNS_BASE

    run_ecj_like_experiment_from_runs(
        feature_type=args.feature_type,
        n_features=n_features,
        base_runs_dir=base_runs_dir,
        num_total_jobs=config.NUM_JOBS, # 5 jobs
        num_runs_per_job=config.N_SPLITS # 10 runs/folds per job
    )