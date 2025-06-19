from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np


class ClassificationStandard:
    def __init__(self, training_set, test_set):
        self.training_set = training_set
        self.test_set = test_set
        self.current_sample = None

    def evaluate(self, individual, state=None):
        """评估个体适应度"""
        # 转换训练集特征
        X_train, y_train = self.transform_dataset(individual, self.training_set)

        # 转换测试集特征
        X_test, y_test = self.transform_dataset(individual, self.test_set)

        # 评估分类器
        svm_score = self.evaluate_classifier(SVC(), X_train, y_train, X_test, y_test)
        dt_score = self.evaluate_classifier(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)
        rf_score = self.evaluate_classifier(
            RandomForestClassifier(n_estimators=5, max_depth=5),
            X_train, y_train, X_test, y_test
        )

        # 取最佳分类器得分
        fitness = max(svm_score, dt_score, rf_score)
        return fitness

    def transform_dataset(self, individual, dataset):

        X, y = [], []
        for sample in dataset:
            self.current_sample = sample
            # 执行GP树计算
            result = self.execute_tree(individual[0])
            # 组合原始特征和GP计算结果
            features = sample.features + [result]
            X.append(features)
            y.append(sample.class_label)
        return np.array(X), np.array(y)

    def execute_tree(self, tree):

        return tree.execute(self.current_sample.features)

    def evaluate_classifier(self, clf, X_train, y_train, X_test, y_test):
        """评估单个分类器"""
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        return f1_score(y_test, pred, average='macro')