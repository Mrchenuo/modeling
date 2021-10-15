from sklearn.linear_model import Ridge
from sklearn.svm import SVR  # SVM中的回归算法
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, metrics


def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')

# 获取数据
train_df = pd.read_excel("../data/train.xlsx", engine='openpyxl', sheet_name='Sheet1')
train_df = train_df.iloc[:-2, 1:]  # 最后两行为nan去掉，第一列为名称去掉
var_top20 = np.load(r'../feature_extraction/variable_top20.npy')
var_top20 = pd.Index(var_top20.tolist())
train_df = train_df[var_top20]
X = train_df.iloc[:, 1:].values
Y = train_df.iloc[:, 0].values

# 将每个特征值归一化到一个固定范围
# 原始数据标准化，为了加速收敛
# 最小最大规范化对原始数据进行线性变换，变换到[0,1]区间
data_X = preprocessing.MinMaxScaler().fit_transform(X)

# 利用train_test_split 进行训练集和测试集进行分开
X_train, X_test, y_train, y_test = train_test_split(data_X, Y, test_size=0.3)

# 通过多种模型预测
# # SVM
# model_svr1 = SVR(kernel='rbf', C=50, max_iter=10000)
# # 训练
# # model_svr1.fit(data_X,Y)
# model_svr1.fit(X_train, y_train)
# preds = model_svr1.predict(X_test)
# print_evaluate(y_test, preds)
# # 得分
# score = model_svr1.score(X_test, y_test)
# print(score)

# ridge
model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
print("用训练好的模型在验证集上预测")
preds = model.predict(X_test)
print_evaluate(y_test, preds)

