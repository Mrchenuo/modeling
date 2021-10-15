import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



if __name__ == '__main__':
    # 从硬盘读取数据进入内存
    data_train_df = pd.read_excel("../data/Molecular_Descriptor.xlsx", engine='openpyxl', sheet_name='Sheet1')
    label_train_df = pd.read_excel("../data/ERα_activity.xlsx", engine='openpyxl', sheet_name='Sheet1')
    # x为除了SMILES外的729个变量，y为pIC50
    x, y = data_train_df.iloc[:, 1:].values, label_train_df.iloc[:, 2].values

    y = y * 1000
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    feat_labels = data_train_df.columns.tolist()[1:]
    forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    forest.fit(x_train, y_train.astype('int'))

    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
