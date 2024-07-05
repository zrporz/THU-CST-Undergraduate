import pdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedKFold
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
bool_post_code_interst_mean= False
industry_mean_interest= False
bool_recircle_u_b_std = False
bool_early_return_amount_early_return =True
bool_total_loan_monthly_payment = False
from random_forest import RandomForest
import time
import argparse

def preprocess():    
    train_data = pd.read_csv('train_public.csv')
    test_public = pd.read_csv('test_public.csv')
    train_inte = pd.read_csv('train_internet.csv')
    train_inte = train_inte.dropna()
    
    # 缺失值补充
    cols_to_fill = ['recircle_u', 'pub_dero_bankrup', 'debt_loan_ratio', 'f0', 'f1', 'f2', 'f3', 'f4']
    train_data[cols_to_fill] = train_data[cols_to_fill].fillna(train_data[cols_to_fill].median())
    test_public[cols_to_fill] = test_public[cols_to_fill].fillna(test_public[cols_to_fill].median())
    train_data['post_code'] = train_data['post_code'].fillna(train_data['post_code'].mode()[0])
    test_public['post_code'] = test_public['post_code'].fillna(test_public['post_code'].mode()[0])

    def parse_work_years(x):
        if pd.isnull(x) or '< 1' in str(x):
            return 0
        else:
            numbers = re.findall(r'\d+', str(x))
            return int(numbers[0]) if numbers else 0

    train_data['work_year'] = train_data['work_year'].map(parse_work_years)
    test_public['work_year'] = test_public['work_year'].map(parse_work_years)
    train_inte['work_year'] = train_inte['work_year'].map(parse_work_years)
    
    class_dict = {
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': 7,
    }
    train_data['class'] = train_data['class'].map(class_dict)
    test_public['class'] = test_public['class'].map(class_dict)
    train_inte['class'] = train_inte['class'].map(class_dict)


    cat_cols = ['employer_type', 'industry']
    from sklearn.preprocessing import LabelEncoder

    for col in cat_cols:
        lbl = LabelEncoder().fit(train_data[col])
        train_data[col] = lbl.transform(train_data[col])
        test_public[col] = lbl.transform(test_public[col])
        train_inte[col] = lbl.transform(train_inte[col])

    print(train_inte['sub_class'].unique())
    print(train_data['class'].unique())
    print(test_public['class'].unique())
    '''
    ['A1' 'A2' 'A3' 'A4' 'A5' 'B1' 'B2' 'B3' 'B4' 'B5' 'C1' 'C2' 'C3' 'C4' 'C5' 'D1' 'D2' 'D3' 'D4' 'D5' 'E1' 'E2' 
    'E3' 'E4' 'E5' 'F1' 'F2' 'F3' 'F4' 'F5' 'G1' 'G2' 'G3' 'G4' 'G5']
    共七大类其中每大类中有五小类，采用聚类方案
    '''

    # 获取特征列表
    train_data_feats = set(train_data.columns)
    train_inte_feats = set(train_inte.columns)

    # 找出相同的特征
    common_feats = train_data_feats & train_inte_feats

    # 在相同的特征中，去除不需要的特征
    excluded_feats = {'loan_id', 'user_id', 'issue_date', 'earlies_credit_mon', 'isDefault', 'class', 'sub_class'}
    training_feats = list(common_feats - excluded_feats)

    # 对 train_data 和 test_data 进行分类
    def classify_subclass(data, classifier, scaler, training_feats):
        X_test = scaler.transform(data[training_feats])
        y_pred = classifier.predict(X_test)
        # pdb.set_trace()
        return pd.Series(y_pred, index=data.index)
    train_preds = []
    test_preds = []
    # 预测 sub_class
    for label in range(1, 8):  # 假设 class 为 1 到 7
        train_inte_class = train_inte[train_inte['class'] == label]
        test_data_class = test_public[test_public['class'] == label]
        train_data_class = train_data[train_data['class'] == label]
        # 数据标准化
        sscaler = StandardScaler()
        print("transforming data!")
        X_train_inte = sscaler.fit_transform(train_inte_class[training_feats])
        y_train_inte = train_inte_class['sub_class']
        # pdb.set_trace()
        # 初始化分类器
        classifier = RandomForestClassifier(random_state=42, n_estimators=100)
        print("Begin training!")
        # 训练分类器
        classifier.fit(X_train_inte, y_train_inte)
        print("predict train")
        train_pred = classify_subclass(train_data_class, classifier, sscaler, training_feats)
        train_preds.append(train_pred)
        print("predict test")
        test_pred = classify_subclass(test_data_class, classifier, sscaler, training_feats)
        test_preds.append(test_pred)

        train_data['sub_class'] = pd.concat(train_preds).reset_index(drop=True)
        test_public['sub_class'] = pd.concat(test_preds).reset_index(drop=True)

    cat_cols = ['sub_class']
    for col in cat_cols:
        lbl = LabelEncoder().fit(train_data[col])
        train_data[col] = lbl.transform(train_data[col])
        test_public[col] = lbl.transform(test_public[col])
        train_inte[col] = lbl.transform(train_inte[col])

    ######### 保存这些特征 #########
    save_new_train = 'train_data_new.csv'
    save_new_test = 'test_public_new.csv'
    train_data.to_csv(save_new_train, index=False)
    test_public.to_csv(save_new_test, index=False)
    print(f"Features add subclass, save to {save_new_train} and {save_new_test}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--method', type=str, choices=['RF', 'XGB', "MLP"], help="选择方法")
    args = parser.parse_args()
    if args.preprocess:
        preprocess()
    if args.method == 'XGB':
        
        ######### 重新加载这些特征 #########
        train_data = pd.read_csv('train_data_new.csv')
        test_public = pd.read_csv('test_public_new.csv')


        #######尝试新的特征######################
        if bool_early_return_amount_early_return:
            train_data['early_return_amount_early_return'] = train_data['early_return_amount'] / train_data['early_return']
            test_public['early_return_amount_early_return'] = test_public['early_return_amount'] / test_public['early_return']
            # 可能出现极大值和空值
            train_data['early_return_amount_early_return'][np.isinf(train_data['early_return_amount_early_return'])] = 0
            test_public['early_return_amount_early_return'][np.isinf(test_public['early_return_amount_early_return'])] = 0

            train_data['early_return_amount_3mon_early_return'] = train_data['early_return_amount_3mon'] / train_data['early_return']
            test_public['early_return_amount_3mon_early_return'] = test_public['early_return_amount_3mon'] / test_public['early_return']
            # 可能出现极大值和空值
            train_data['early_return_amount_3mon_early_return'][np.isinf(train_data['early_return_amount_3mon_early_return'])] = 0
            test_public['early_return_amount_3mon_early_return'][np.isinf(test_public['early_return_amount_3mon_early_return'])] = 0

            train_data['f3*f4'] = train_data['f3'] * train_data['f4']
            test_public['f3*f4'] = test_public['f3'] * test_public['f4']
            
        y = train_data['isDefault']
        
        test_data = test_public
        print(train_data.columns)
        # 应用预处理
        train_trimmed_data = train_data.drop(['isDefault','loan_id', 'user_id'], axis=1)
        test_trimmed_data = test_data.drop(['loan_id', 'user_id'], axis=1)

        # 预处理步骤
        numerical_cols = train_trimmed_data.select_dtypes(include=['float64','float32','int32', 'int64']).columns
        categorical_cols = test_trimmed_data.select_dtypes(include=['object']).columns

        # 创建数值和类别数据的转换器
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # 结合转换器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        X = preprocessor.fit_transform(train_trimmed_data)
        y = train_data['isDefault'].values
        X_test = preprocessor.transform(test_trimmed_data)

        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        from xgboost import XGBClassifier
        # 训练XGBoost模型
        # xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        from lightgbm import LGBMClassifier
        import gc
        folds = KFold(n_splits=5, shuffle=True, random_state=546789)
        sub_preds = np.zeros(X_test.shape[0])

        val_auc_list = []
        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X)):
            xgbmodel = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
            trn_x, trn_y = X[trn_idx], y[trn_idx]
            val_x, val_y = X[val_idx], y[val_idx]
            # import ipdb; ipdb.set_trace()

            xgbmodel.fit(trn_x, trn_y,eval_set=[(trn_x, trn_y.astype(int)), (val_x, val_y.astype(int))],
                        verbose=False   )  # 40)
            sub_preds += xgbmodel.predict_proba(X_test)[:, 1] / folds.n_splits

            # 验证模型
            y_val_pred = xgbmodel.predict_proba(val_x)[:, 1]
            val_auc = roc_auc_score(val_y, y_val_pred)
            val_auc_list.append(val_auc)
            print(f'Fold {n_fold+1}, Validation AUC: {val_auc}')

        # 预测测试集
        test_preds = sub_preds

        # 准备提交文件
        submission = pd.DataFrame({
            'id': test_data['loan_id'],
            'isDefault': test_preds
        })
        save_name = f'auc={(sum(val_auc_list)/len(val_auc_list))*100:.3f}.csv'
        submission.to_csv(save_name, index=False)

        # 输出路径
        print(save_name)
        
    elif args.method == 'RF':
        # 加载数据
        train_data = pd.read_csv('train_public.csv')
        test_data = pd.read_csv('test_public.csv')

        # 预处理步骤
        numerical_cols = train_data.select_dtypes(include=['float64', "int32", "float32", 'int64']).columns.drop('isDefault')

        # 数据标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(train_data[numerical_cols])
        y = train_data['isDefault'].values
        X_test = scaler.transform(test_data[numerical_cols])

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 定义随机森林模型
        model = RandomForest(n_estimators=15, max_depth=15, min_samples_split=2, seed_value=42)

        # 训练模型
        start_time = time.time()
        model.fit(X_train, y_train)
        print(f"Training time: {time.time() - start_time} seconds")

        # 验证模型
        y_val_pred = model.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, y_val_pred)
        print(f'Validation AUC: {val_auc}')

        # 预测测试集
        test_preds = model.predict_proba(X_test)

        # 准备提交文件
        submission = pd.DataFrame({
            'id': test_data['loan_id'],
            'isDefault': test_preds
        })
        submission.to_csv('predictions.csv', index=False)

        # 输出路径
        print('predictions.csv')
        
    elif args.method == 'MLP':
        train_data = pd.read_csv('train_public.csv')
        test_data = pd.read_csv('test_public.csv')
        numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.drop('isDefault')
        categorical_cols = train_data.select_dtypes(include=['object']).columns

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        print(train_data)
        trimmed_data = train_data.drop(['isDefault'], axis=1)
        print("======================")
        X_train = preprocessor.fit_transform(trimmed_data)
        y_train = train_data['isDefault'].values
        X_test = preprocessor.transform(test_data)
        X_test_tensor = torch.tensor(X_test.toarray().astype(np.float32))

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # 转换为张量
        X_train_tensor = torch.tensor(X_train.toarray().astype(np.float32))
        y_train_tensor = torch.tensor(y_train.astype(np.int64))
        X_val_tensor = torch.tensor(X_val.toarray().astype(np.float32))
        y_val_tensor = torch.tensor(y_val.astype(np.int64))

        # DataLoader
        train_data_tensor = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data_tensor, batch_size=64, shuffle=True)

        val_data_tensor = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_data_tensor, batch_size=64, shuffle=False)

        # 定义MLP模型
        class MLP(nn.Module):
            def __init__(self, num_features):
                super(MLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(num_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                return self.layers(x)

        # 初始化模型
        model = MLP(X_train_tensor.shape[1])
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 转换数据为DataLoader
        train_data_tensor = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data_tensor, batch_size=64, shuffle=True)

        # 训练模型
        # num_epochs = 30
        # for epoch in range(num_epochs):
        #     for inputs, targets in train_loader:
        #         optimizer.zero_grad()
        #         outputs = model(inputs)
        #         loss = loss_function(outputs.squeeze(), targets.float())
        #         loss.backward()
        #         optimizer.step()

        #     print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()  # 设置模型为训练模式
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs.squeeze(), targets.float())
                loss.backward()
                optimizer.step()
            
            # 验证阶段
            model.eval()  # 设置模型为评估模式
            val_loss = 0
            predictions = []
            labels = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss += loss_function(outputs.squeeze(), targets.float()).item()
                    predictions.extend(outputs.squeeze().numpy())
                    labels.extend(targets.numpy())

            val_loss /= len(val_loader)
            val_auc = roc_auc_score(labels, predictions)  # 计算AUC
            print(f'Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss}, Val AUC: {val_auc}')

        # 预测测试集
        with torch.no_grad():
            test_preds = model(X_test_tensor).squeeze()

        # 准备提交文件
        submission = pd.DataFrame({
            'id': test_data['loan_id'],
            'isDefault': test_preds.numpy()
        })
        submission.to_csv('predictions.csv', index=False)

        # 输出路径
        print('predictions.csv')
