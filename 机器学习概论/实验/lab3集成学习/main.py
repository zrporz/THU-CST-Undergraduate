import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score,confusion_matrix
import loguru
import argparse
import pandas as pd
import os.path as osp
import os

def benchmark_test(predictions,y,logger,model_name,result_save_path):
    # import pdb; pdb.set_trace()
    mae = mean_absolute_error(y, predictions)
    rmse = mean_squared_error(y, predictions, squared=False)
    acc = accuracy_score(y, np.round(predictions).astype(int))
    mape = np.mean(np.abs((y - predictions) / y)) * 100
    prec = precision_score(y, np.round(predictions).astype(int), average='macro')
    recall = recall_score(y, np.round(predictions).astype(int), average='macro')
    f1 = f1_score(y, np.round(predictions).astype(int), average='macro')

    logger.info(f"=== {model_name} ===")
    logger.info(f"  MAE: {mae}")
    logger.info(f"  RMSE: {rmse}")
    logger.info(f"  Accuracy: {acc}")
    logger.info(f"  MAPE: {mape}")
    logger.info(f"  Prec: {prec}")
    logger.info(f"  Recall: {recall}")
    logger.info(f"  F1Score: {f1}")
    # 创建混淆矩阵
    conf_matrix = confusion_matrix(y, np.round(predictions).astype(int))
    # 创建热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')

    # 添加标签和标题
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    # 设置横纵坐标轴刻度
    tick_marks = np.arange(len(np.unique(y))) + 1
    plt.xticks(tick_marks, np.unique(y))
    plt.yticks(tick_marks, np.unique(y))
    # 显示图像
    plt.savefig(osp.join(result_save_path, f"{model_name}_conf_matrix.png"))
    return {
        "MAE":round(mae,2),
        "RMSE":round(rmse,2),
        "Accuracy":round(acc,2),
        "MAPE":round(mape,2),
        "Prec":round(prec,2),
        "Recall":round(recall,2),
        "F1Score":round(f1,2)
    }


class MyBagging:
    def __init__(self,base_estimator,n_base):
        assert base_estimator=="SVM" or base_estimator=="SCTree", "Base estimator must be SVM or DecisionTree"
        self.base_estimator = base_estimator
        self.estimators_list = []
        self.n_base = n_base

    def fit(self,X,y):
        for i in tqdm(range(self.n_base)):
            indices = np.random.choice(np.arange(len(X)),int(0.1*len(X)),replace=True)
            if self.base_estimator == "SVM":
                estimator = SVC()
            else:
                estimator = DecisionTreeClassifier()
            estimator.fit(X[indices], y[indices])
            self.estimators_list.append(estimator)
    
    def predict(self,X):
    
        predictions = np.zeros((len(X), len(self.estimators_list)))
        for i, estimator in tqdm(enumerate(self.estimators_list)):
            predictions[:, i] = estimator.predict(X)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions.astype(int)).flatten()

class MyBaggingRegressor:
    def __init__(self,base_estimator,n_base):
        assert base_estimator=="SVR" or base_estimator=="SCTree", "Base estimator must be SVM or DecisionTree"
        self.base_estimator = base_estimator
        self.estimators_list = []
        self.n_base = n_base

    def fit(self,X,y):
        for _ in tqdm(range(self.n_base)):
            indices = np.random.choice(np.arange(len(X)),int(0.1*len(X)),replace=True)
            if self.base_estimator == "SVR":
                estimator = SVR(kernel='linear', C=1.0, epsilon=0.1)
            else:
                estimator = DecisionTreeRegressor(random_state=np.random.randint(0,1000))
            estimator.fit(X[indices], y[indices])
            self.estimators_list.append(estimator)

    def predict(self,X):
        predictions = np.zeros((len(X), len(self.estimators_list)))
        for i, estimator in tqdm(enumerate(self.estimators_list)):
            predictions[:, i] = estimator.predict(X)
        return np.mean(predictions, axis=1)
    
class MyAdaBoost:
    def __init__(self, base_estimator, n_base):
        assert base_estimator=="SVM" or base_estimator=="SCTree", "Base estimator must be SVM or DecisionTree"
        self.base_estimator = base_estimator
        self.estimators_list = []
        self.alphas = []
        self.n_base = n_base
        self.last_error = None

    def fit(self, X, y):
        n_samples = len(X)
        weights = np.ones(n_samples) / n_samples  # Initialize weights uniformly
        
        for _ in tqdm(range(self.n_base)):
            if self.base_estimator == "SVM":
                estimator = SVC(random_state=np.random.randint(0,1000))
            else:
                estimator = DecisionTreeClassifier(random_state=np.random.randint(0,1000))
            estimator.fit(X, y)
            predictions = estimator.predict(X)
            error = np.sum(weights * (predictions != y)) / np.sum(weights)  # Weighted error
            # Calculate alpha
            if error == 0:
                last_alpha = self.alphas[-1] if self.alphas else 1
                self.alphas.append(last_alpha)
                continue
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            
            # Update weights
            weights *= np.exp(np.where(y == predictions, -alpha, alpha))
            # if  np.sum(weights)==0:
            #     import pdb; pdb.set_trace()
            weights /= np.sum(weights)
            self.estimators_list.append(estimator)
            self.last_error = error
        
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for alpha, estimator in zip(self.alphas, self.estimators_list):
            predictions += alpha * estimator.predict(X)
        predictions/=sum(self.alphas)
        return np.where(predictions >0.5, 1, 0)
        # return np.sign(predictions)



class MyAdaBoostRegressor:
    def __init__(self, base_estimator, n_base):
        assert base_estimator=="SVR" or base_estimator=="SCTree", "Base estimator must be SVM or DecisionTree"
        self.base_estimator = base_estimator
        self.estimators_list = []
        self.alphas = []
        self.n_base = n_base
        self.last_error = None

    def fit(self, X, y):
        n_samples = len(X)
        weight = np.ones(n_samples) / n_samples  # Initialize weights uniformly
        
        for _ in tqdm(range(self.n_base)):
            # self.weights.appedn(weight)
            if self.base_estimator == "SVR":
                estimator = SVR(kernel='linear', C=1.0, epsilon=0.1)
            else:
                estimator = DecisionTreeRegressor(random_state=np.random.randint(0,1000))
            estimator.fit(X, y)
            predictions = estimator.predict(X)
            errors = np.abs(predictions - y)  # Absolute errors
            errors = errors/errors.max()
            total_error = np.sum(weight * errors)  # Weighted total error
            if total_error == 0:
                last_alpha = self.alphas[-1] if self.alphas else 1
                self.alphas.append(last_alpha)
                continue
            # Avoid division by zero
            # if total_error == 0:
            #     alpha = 1
            # else:
            alpha = 0.5 * np.log((1-total_error)/total_error)
            self.alphas.append(alpha)
            
            # Update weights
            weight *= np.exp(-alpha*errors)
            weight /= np.sum(weight)
            self.estimators_list.append(estimator)
            # self.last_error = total_error
        
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for alpha, estimator in zip(self.alphas, self.estimators_list):
            # print(predictions)
            # print(alpha)
            predictions += alpha * estimator.predict(X)
        return predictions/sum(self.alphas)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_estimator", type=int,default=10)
    parser.add_argument("--train_ratio", type=float,default=0.01)
    parser.add_argument("--max_features", type=int,default=100)
    parser.add_argument("--seed", type=int,default=42)
    parser.add_argument("--vectorizer", choices=["Tfidf","Count"],default="Tfidf")
    args = parser.parse_args()
    result_save_path = osp.join("output",f"n{args.num_estimator}-r{args.train_ratio}-f{args.max_features}-v{args.vectorizer}")
    os.makedirs(result_save_path,exist_ok=True)
    data = pd.read_csv('exp3-reviews.csv',sep='\t')  # 替换为你的数据集路径
    logger = loguru.logger
    logger.add("log.log", rotation="10 MB", retention="10 days", level="INFO")
    num_estimator = args.num_estimator
    train_ratio = args.train_ratio
    max_features = args.max_features
    np.random.seed(seed=args.seed)
    # 示例数据和特征提取
    vectorizer = TfidfVectorizer(stop_words='english',max_features=max_features) if args.vectorizer == "Tfidf" else CountVectorizer(stop_words='english',max_features=max_features)
    # vector_list = [CountVectorizer(stop_words='english',max_features=max_features) ,TfidfVectorizer(stop_words='english',max_features=max_features)]
    # for vectorizer in vector_list:
    all_results = []
    logger.info("Starting...")
    logger.info(f"Setting, {args}")
    print("Extracting features...")
    X = vectorizer.fit_transform(data['reviewText']).toarray()
    y = data['overall'].values.astype(int)
    print("Done")
    # import pdb; pdb.set_trace()

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=args.seed)
    sample_indices = np.random.choice(np.arange(len(X_train)), size=int(len(X_train)*train_ratio), replace=True)

    ######################### Adaboost #########################
    svm_ada_list = []
    dt_ada_list = []
    # 使用SVM和决策树作为基分类器
    for label in range(1,6):
        svm_ada = MyAdaBoost("SVM", n_base=num_estimator)
        dt_ada = MyAdaBoost("SCTree", n_base=num_estimator)
        # sample_indices = np.random.choice(np.arange(len(X_train)), size=int(len(X_train)*train_ratio), replace=True)
        X_train_transformed = X_train[sample_indices]
        y_train_transformed = np.where(y_train[sample_indices] == np.ones(len(sample_indices))*label, 1, 0)
        # 训练模型
        # import pdb; pdb.set_trace()
        svm_ada.fit(X_train_transformed, y_train_transformed)
        dt_ada.fit(X_train_transformed, y_train_transformed)
        svm_ada_list.append(svm_ada)
        dt_ada_list.append(dt_ada)

    svm_predict_candidate = np.zeros((len(X_test), 5))
    dt_predict_candidate = np.zeros((len(X_test), 5))
    # 评估模型
    for label in range(1,6):
        svm_ada = svm_ada_list[label-1]
        dt_ada = dt_ada_list[label-1]
        
        # 预测测试集
        svm_ada_predictions = svm_ada.predict(X_test)
        dt_ada_predictions = dt_ada.predict(X_test)

        svm_predict_candidate[:,label-1]= svm_ada_predictions
        dt_predict_candidate[:,label-1]= dt_ada_predictions
        # y_test_transformed= np.where(y_test == np.ones(len(y_test))*label, 1, 0)

        # benchmark_test(svm_ada_predictions,y_test_transformed,logger,f"SVM AdaBoost {label}")
        # benchmark_test(dt_ada_predictions,y_test_transformed,logger,f"DecetionTree AdaBoost {label}")

    # 随机选择一个标签
    def random_choice(predictions):
        candidates = np.where(predictions == 1)[0]
        if len(candidates) > 0:
            return np.random.choice(candidates) + 1  # 加一是因为 label 是从 1 开始的
        else:
            return np.random.randint(1, 6)  # 如果没有候选标签，则随机选择一个标签
    final_svm_predictions = np.apply_along_axis(random_choice, 1, svm_predict_candidate)
    final_dt_predictions = np.apply_along_axis(random_choice, 1, dt_predict_candidate)

    benchmark_result = benchmark_test(final_svm_predictions,y_test,logger,f"Final SVM AdaBoost",result_save_path)
    benchmark_result["Model"]="Adaboost-SVM"
    all_results.append(benchmark_result)
    benchmark_result = benchmark_test(final_dt_predictions,y_test,logger,f"Final DecetionTree AdaBoost",result_save_path)
    benchmark_result["Model"]="Adaboost-DCT"
    all_results.append(benchmark_result)

    ######################### Adaboost(REG) #########################
    svm_ada = MyAdaBoostRegressor("SVR", n_base=num_estimator)
    dt_ada = MyAdaBoostRegressor("SCTree", n_base=num_estimator)
    
    # 训练模型
    svm_ada.fit(X_train[sample_indices], y_train[sample_indices])
    dt_ada.fit(X_train[sample_indices], y_train[sample_indices])

    # 评估模型
    svm_predictions = svm_ada.predict(X_test)
    dt_predictions = dt_ada.predict(X_test)

    benchmark_result = benchmark_test(svm_predictions,y_test,logger,f"SVR AdaBoostRegressor",result_save_path)
    benchmark_result["Model"] = "AdaboostReg-SVR"
    all_results.append(benchmark_result)
    benchmark_result = benchmark_test(dt_predictions,y_test,logger,f"DecetionTree AdaBoostRegressor",result_save_path)
    benchmark_result["Model"] = "AdaboostReg-DCT"
    all_results.append(benchmark_result)

    ######################### Bagging #########################
    # 使用SVM和决策树作为基分类器
    svm_bagging = MyBagging("SVM", n_base=num_estimator)
    dt_bagging = MyBagging("SCTree", n_base=num_estimator)

    # 训练模型
    svm_bagging.fit(X_train[sample_indices], y_train[sample_indices])
    dt_bagging.fit(X_train[sample_indices], y_train[sample_indices])

    # 评估模型
    svm_predictions = svm_bagging.predict(X_test)
    dt_predictions = dt_bagging.predict(X_test)

    benchmark_result = benchmark_test(svm_predictions,y_test,logger,f"SVM Bagging",result_save_path)
    benchmark_result["Model"] = "Bagging-SVM"
    all_results.append(benchmark_result)
    benchmark_result = benchmark_test(dt_predictions,y_test,logger,f"DecetionTree Bagging",result_save_path)
    benchmark_result["Model"] = "Bagging-DCT"
    all_results.append(benchmark_result)

    ######################### Bagging(REG) #########################

    # 使用SVM和决策树作为基分类器
    svm_bagging = MyBaggingRegressor("SVR", n_base=num_estimator)
    dt_bagging = MyBaggingRegressor("SCTree", n_base=num_estimator)

    # 训练模型
    svm_bagging.fit(X_train[sample_indices], y_train[sample_indices])
    dt_bagging.fit(X_train[sample_indices], y_train[sample_indices])

    # 评估模型
    svm_predictions = svm_bagging.predict(X_test)
    dt_predictions = dt_bagging.predict(X_test)

    benchmark_result = benchmark_test(svm_predictions,y_test,logger,f"SVM Bagging Reg",result_save_path)
    benchmark_result["Model"] = "BaggingReg-SVR"
    all_results.append(benchmark_result)
    benchmark_result = benchmark_test(dt_predictions,y_test,logger,f"DecetionTree Bagging Reg",result_save_path)
    benchmark_result["Model"] = "BaggingReg-DCT"
    all_results.append(benchmark_result)
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_results)

    # Set the index column
    df.set_index("Model", inplace=True)

    # Export to CSV
    df.to_csv(osp.join(result_save_path,"results.csv"))
    logger.info(f"result saved in {result_save_path}")