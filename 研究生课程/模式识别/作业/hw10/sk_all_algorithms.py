from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集并进行train/val/test划分
datasets = {
    "iris": load_iris(),
    "breast_cancer": load_breast_cancer(),
    "usps": fetch_openml('usps', version=3)
}

X_train_sets, X_val_sets, X_test_sets, y_train_sets, y_val_sets, y_test_sets = {}, {}, {}, {}, {}, {}
scaler = StandardScaler()

# print("Loading datasets")
for name, data in datasets.items():
    # print(f"=> Loading {name}")
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, stratify=y_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_sets[name], X_val_sets[name], X_test_sets[name], y_train_sets[name], y_val_sets[name], y_test_sets[name] = X_train_scaled, X_val_scaled, X_test_scaled, y_train.astype(int), y_val.astype(int), y_test.astype(int)
    # print("=> Done!")
# print("Dataset loading complete.")

# 这里我们以Decision Tree为例，定义了模型以及其超参数（criterion选什么以及max_depth怎么取），请大家完成其他模型的定义以及超参数设置
models = {
    "Decision Tree": DecisionTreeClassifier,  # criterion, max_depth
    ##################################################
    # TODO: 找到下面每个算法对应的sklearn模型
    ##################################################
    # "Random Forest":
    # "Bagging":
    # "Gradient Boosting":
    # "XGBoost":
    # "Naive Bayes":
    # "Perceptron":
    # "Logistic Regression":
    # "LDA":
    # "SVM":
    "Random Forest": RandomForestClassifier,
    "Bagging": BaggingClassifier,
    # "Gradient Boosting": GradientBoostingClassifier,
    # "XGBoost": XGBClassifier,
    # "Naive Bayes": GaussianNB,
    # "Perceptron": Perceptron,
    # "Logistic Regression": LogisticRegression,
    # "LDA": LinearDiscriminantAnalysis,
    # "SVM": SVC,
}
import os
model_hparams = {
    "Decision Tree": dict(criterion='gini', max_depth=5),
    ##################################################
    # TODO: 为下面每个模型指定超参数
    ##################################################
    "Random Forest": dict(criterion='gini', n_estimators=100),
    #  "Random Forest": dict(criterion='log_loss', n_estimators=50),
    "Bagging":dict(n_estimators=20),
    # "Gradient Boosting":dict(n_estimators=200, learning_rate=1e-1),
    # "Naive Bayes":dict(var_smoothing=1e-1),
    # "Perceptron":dict(penalty='l1',alpha=1e-3),
    # "Logistic Regression":dict(penalty='l2',C=10),
    #  "LDA":dict(),
    #  "SVM":dict(kernel='linear',C=1),
}

def run_method(method_name):
    # 这个函数用于调试某一个特定方法的超参数，并在训练、验证集上测试准确率
    train_results, val_results,test_results = {}, {}, {}
    hparams = model_hparams.get(method_name, dict())
    for name, X_train, X_val, X_test, y_train, y_val, y_test in zip(
            datasets.keys(), X_train_sets.values(), X_val_sets.values(),X_test_sets.values(), y_train_sets.values(), y_val_sets.values(), y_test_sets.values()
    ):
        model_fn = models[method_name]
        model = model_fn(**hparams)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        test_score = model.score(X_test, y_test)
        train_results[name] = train_score
        val_results[name] = val_score
        test_results[name] = test_score
        
    return train_results, val_results,test_results, hparams

def report_result(runs = 5):
    # 这个函数将每个算法都运行多次，用于减少随机误差。并汇总了每个算法在所有数据集的测试集上的结果，以柱状图形式呈现最终表现。
    results = []
    for name, X_train, X_test, y_train, y_test in zip(
        datasets.keys(), X_train_sets.values(), X_test_sets.values(), y_train_sets.values(), y_test_sets.values()
    ):
        row = {"Dataset": name}

        for model_name, model_fn in models.items():
            score = 0
            for _ in range(runs):
                model = model_fn(**model_hparams.get(model_name, dict()))
                model.fit(X_train, y_train)
                score += model.score(X_test, y_test)
            row[model_name] = score / runs
        results.append(row)

    df = pd.DataFrame(results)
    print(df)
    # df_melt = pd.melt(df, id_vars=["Dataset"], var_name="Algorithm", value_name="Accuracy")

    # sns.catplot(x="Dataset", y="Accuracy", hue="Algorithm", data=df_melt, kind="bar", height=6, aspect=1.5)
    # plt.ylim(0.8, 1.0)
    # plt.xlabel("Dataset")
    # plt.ylabel("Accuracy")
    # plt.show()


if __name__ == "__main__":
    acc = 0
    for _ in range(5):
        train_results, val_results,test_results, hparams = run_method('Bagging')
        acc += val_results['usps']
        # f'train accuracy on each dataset: {train_results}'
    print(
          f'\nval accuracy on each dataset: {acc*20:.2f}'
          f'\nhparams: {hparams}')
        #   f'\ntest accuracy on each dataset: {test_results}'
    report_result()
   