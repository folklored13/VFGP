import random
import numpy as np
import os
from deap import base, creator, tools, gp, algorithms
from deap.gp import PrimitiveSetTyped

import functions as fs
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import math

# Individual, Fitness, and Toolbox setup (as in original code)
creator.create("FeatureVectorType", list) # Keep this for the input vector type
creator.create("FeatureIndex", int)      # Define a type for integer indices
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


data_path = "E:/datasets/PH2-dataset/processed_ph2/"
x_train = np.load(os.path.join(data_path, "train_features.npy"))
y_train = np.load(os.path.join(data_path, "train_labels.npy"))
x_test = np.load(os.path.join(data_path, "test_features.npy"))
y_test = np.load(os.path.join(data_path, "test_labels.npy"))

# Scale data (as in original code)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define GP tree types: Input is the full feature vector (list of floats), Output is a single float (constructed feature)
FeatureVector = list # Type hint for the input vector
Float = float # Type hint for scalar values

# Create the Primitive Set for Typed GP
# Input ARG0 is FeatureVector (the full original feature vector for a sample)
# The root of the tree returns a Float (the constructed feature value)
pset = PrimitiveSetTyped("MAIN", [creator.FeatureVectorType], Float)
pset.renameArguments(ARG0='original_features')

# Add Terminals: One terminal for each original feature
# The type of the terminal is Float, as it represents a single scalar feature value
for i in range(x_train.shape[1]):
    # The terminal value is the integer 'i', the type is FeatureIndex
    pset.addTerminal(i, creator.FeatureIndex, name=f"F{i}")
pset.addTerminal(0.0, Float, name="Const0")
pset.addTerminal(1.0, Float, name="Const1")
# Add Primitives: Operators work on Floats and return Floats
pset.addPrimitive(fs.add_s, [Float, Float], Float, name="add")
pset.addPrimitive(fs.sub_s, [Float, Float], Float, name="sub")
pset.addPrimitive(fs.mul_s, [Float, Float], Float, name="mul")
pset.addPrimitive(fs.protectedDiv_s, [Float, Float], Float, name="div") # Use scalar protected div
pset.addPrimitive(fs.sin_s, [Float], Float, name="sin")
pset.addPrimitive(fs.cos_s, [Float], Float, name="cos")
pset.addPrimitive(fs.if_s, [Float, Float, Float, Float], Float, name="If") # Use scalar If
pset.addPrimitive(fs.get_feature_value_func, [creator.FeatureVectorType, creator.FeatureIndex], Float, name="GetValue")

toolbox = base.Toolbox()
# Use genHalfAndHalf with correct pset and depth limits from paper (Table IV)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# --- Define Evaluation Function ---
def evalVFGP(individual):
    try:
        # Compile the GP tree into a callable function
        # This function takes the full feature vector (as defined in pset input)
        # and returns a single float (as defined in pset output)
        gp_compiled_func = toolbox.compile(expr=individual)

        # --- Construct the NEW variable-length feature vector for the training data (x_train) ---

        # 1. Get the indices of the selected features (terminals of the current individual tree)
        # Use a set to get unique indices, then sort them for consistent ordering
        selected_indices = sorted(list({node.value for node in individual if isinstance(node, gp.Terminal)}))

        # 2. Create the new transformed training dataset
        X_transformed_cv_list = []
        for sample in x_train:
            # Get the values of the selected features for this sample
            selected_values = sample[selected_indices]

            # Calculate the constructed feature value for this sample using the compiled GP tree
            # The compiled function takes the full original sample (as list) as input
            constructed_value = gp_compiled_func(sample.tolist())

            # Concatenate selected features and the constructed feature
            transformed_sample = np.concatenate((selected_values, [constructed_value])) # [constructed_value] makes it a 1-element array to concatenate

            X_transformed_cv_list.append(transformed_sample)

        # Convert the list of transformed samples into a NumPy array
        X_transformed_cv_np = np.array(X_transformed_cv_list)


        # --- Evaluate the ensemble classifier using 10-fold stratified cross-validation ---
        # Train and evaluate on the NEW variable-length feature vector

        clf1 = SVC(kernel='rbf', probability=True, random_state=42)
        clf2 = DecisionTreeClassifier(criterion='entropy', random_state=42) # J48 is a variant of C4.5/CART, often implemented by DecisionTreeClassifier
        clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
        # Note: VotingClassifier trains base estimators on the *same* data internally during fit
        eclf = VotingClassifier(estimators=[('svm', clf1), ('j48', clf2), ('rf', clf3)], voting='soft')

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # 10-fold stratified CV from paper (IV.D)

        f1_scores = []
        # Use the transformed training data and original labels for CV
        for train_idx, val_idx in skf.split(X_transformed_cv_np, y_train):
            X_train_fold, X_val_fold = X_transformed_cv_np[train_idx], X_transformed_cv_np[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Train the ensemble on the training fold of the transformed data
            eclf.fit(X_train_fold, y_train_fold)
            # Predict on the validation fold of the transformed data
            y_pred = eclf.predict(X_val_fold)

            # Calculate F1 score (macro average) for the validation fold
            # The paper calculates mean of per-class F1, which is macro F1
            f1_macro = f1_score(y_val_fold, y_pred, average='macro') # Use macro average directly
            f1_scores.append(f1_macro)

        # The fitness is the average macro F1 score across all CV folds on the training data
        avg_f1 = np.mean(f1_scores) * 100

    except Exception as e:
        # Catch any errors during compilation, feature construction, or evaluation
        # Assign a very low fitness to invalid individuals
        print(f"Error evaluating individual: {e}")
        avg_f1 = 0.0

    return (avg_f1,) # DEAP fitness must be a tuple

toolbox.register("evaluate", evalVFGP)
# Selection operator from paper (Table IV)
toolbox.register("select", tools.selTournament, tournsize=7)
# Crossover operator from standard GP
toolbox.register("mate", gp.cxOnePoint)
# Mutation operator from standard GP, using genFull with depth limits (e.g., 0-2) for new subtrees
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Main execution block
if __name__ == "__main__":
    # GP Parameters from paper (Table IV)
    population_size = 100
    generations = 50
    cx_prob = 0.8 # Crossover probability
    mut_prob = 0.19 # Mutation probability
    elit_prob = 0.01 # Elitism probability
    # Note: Standard eaSimple doesn't directly use elit_prob, but eaMuPlusLambda does.
    # We can use eaMuPlusLambda to implement selection more closely to the paper.
    mu = population_size
    lambda_ = population_size # Generate lambda offspring (e.g., same as pop size)
    # Select parents for variation using Tournament selection
    selector_parents = toolbox.select
    # Select survivors for the next generation using Best selection (implements elitism)
    num_elites = int(population_size * elit_prob)

    selector_survivors = tools.selBest # Default for mu+lambda survival

    # Initialize population
    pop = toolbox.population(n=population_size)

    # Hall of Fame to store the best individual found
    hof = tools.HallOfFame(1) # Keep only the single best individual

    # Statistics to track during evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    # Add min for logbook completeness
    stats.register("min", np.min)

    # --- Run the GP evolution ---
    # Use algorithms.eaMuPlusLambda for control over parent and survivor selection
    print("Starting GP evolution...")
    pop, log = algorithms.eaMuPlusLambda(
        population=pop,
        toolbox=toolbox,
        mu=mu,         # Number of individuals to select for the next generation
        lambda_=lambda_, # Number of offspring to produce
        cxpb=cx_prob,
        mutpb=mut_prob,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=True
        # selector_parents=selector_parents, # Use Tournament for parents
        # selector_survivors=selector_survivors # Use Best for survivors (mu+lambda)
    )
    print("GP evolution finished.")

    # --- Final Evaluation on the Test Set ---
    # Use the best individual found by GP (from HallOfFame)
    best_individual = hof[0]
    print("\nBest individual:")
    print(best_individual)
    print("Fitness of best individual:", best_individual.fitness.values)

    # Compile the best GP tree
    final_gp_func = toolbox.compile(expr=best_individual)

    # Get the indices of the selected features from the best individual's terminals
    final_selected_indices = sorted(list({node.value for node in best_individual if isinstance(node, gp.Terminal)}))
    print(f"Number of selected features by the best individual: {len(final_selected_indices)}")

    # Construct the final training and test datasets using the best individual
    # New feature vector = [selected_features] + [constructed_feature]

    # Transform Training Data
    X_train_transformed_final_list = []
    for sample in x_train:
        selected_values = sample[final_selected_indices]
        constructed_value = final_gp_func(sample.tolist())
        transformed_sample = np.concatenate((selected_values, [constructed_value]))
        X_train_transformed_final_list.append(transformed_sample)
    X_train_transformed_final_np = np.array(X_train_transformed_final_list)
    print(f"Final transformed training data shape: {X_train_transformed_final_np.shape}")

    # Transform Test Data
    X_test_transformed_final_list = []
    for sample in x_test:
        selected_values = sample[final_selected_indices]
        constructed_value = final_gp_func(sample.tolist())
        transformed_sample = np.concatenate((selected_values, [constructed_value]))
        X_test_transformed_final_list.append(transformed_sample)
    X_test_transformed_final_np = np.array(X_test_transformed_final_list)
    print(f"Final transformed test data shape: {X_test_transformed_final_np.shape}")


    # Train the final ensemble classifier on the transformed training data
    final_clf1 = SVC(kernel='rbf', probability=True, random_state=42)
    final_clf2 = DecisionTreeClassifier(criterion='entropy', random_state=42)
    final_clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
    final_eclf = VotingClassifier(estimators=[('svm', final_clf1), ('j48', final_clf2), ('rf', final_clf3)], voting='soft')

    print("Training final ensemble classifier...")
    final_eclf.fit(X_train_transformed_final_np, y_train)
    print("Final ensemble classifier training finished.")

    # Evaluate the final ensemble classifier on the transformed test data
    print("Evaluating final ensemble classifier on the test set...")
    y_pred_test = final_eclf.predict(X_test_transformed_final_np)

    # Report final results (Accuracy and Macro F1)
    final_accuracy = accuracy_score(y_test, y_pred_test)
    final_f1_macro = f1_score(y_test, y_pred_test, average='macro')

    print("\n=== Final Test Results ===")
    print(f"Test Accuracy: {final_accuracy:.4f}")
    print(f"Test F1 Score (Macro): {final_f1_macro:.4f}")
    print("Best Individual:", best_individual) # Print the best tree structure