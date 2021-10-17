import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

idx = 5
col_name = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
# 1 加载数据
train_X_df = pd.read_excel(r"D:\GitHub\modeling\D\ans3\Molecular_Descriptor.xlsx", engine='openpyxl',
                           sheet_name='training')
train_Y_df = pd.read_excel(r"D:\GitHub\modeling\D\ans3\ADMET.xlsx", engine='openpyxl', sheet_name='training')

X = train_X_df.iloc[:, 1:].values  # 去掉第一列
Y = train_Y_df.iloc[:, idx].values  # 第一个label

nag_sum = np.sum(Y == 0)
pos_sum = np.sum(Y == 1)

# 3 在训练之前，我们需要对数据进行规范化，这样让数据同在同一个量级上，避免因为维度问题造成数据误差：
# 采用 Z-Score 规范化数据，保证每个特征维度的数据均值为 0，方差为 1
ss = StandardScaler()
x_train = ss.fit_transform(X)

model_xgboost = XGBClassifier(booster='gbtree', scale_pos_weight=nag_sum / pos_sum)
model_xgboost.fit(x_train, Y)

# 测试结果
test_X_df = pd.read_excel(r"D:\GitHub\modeling\D\ans3\Molecular_Descriptor.xlsx", engine='openpyxl',
                           sheet_name='test')
test_X = test_X_df.iloc[:, 1:].values  # 去掉第一列
pred = model_xgboost.predict(test_X)
print(pred)

# list转dataframe
# df = pd.DataFrame(pred, columns=[col_name])
# 保存到本地excel
# df.to_excel(col_name + ".xlsx", index=False)

# save model to file
pickle.dump(model_xgboost, open(col_name[idx - 1] + ".pickle", "wb"))
