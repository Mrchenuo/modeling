import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def modelfit(alg, x_train, y_train, x_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(x_test)
    dtrain_predprob = alg.predict_proba(x_test)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)[0:20]
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


# 1 加载数据
X_train_df = pd.read_excel(r"D:\GitHub\modeling\D\ans3\Molecular_Descriptor.xlsx", engine='openpyxl',
                           sheet_name='training')
X = X_train_df.iloc[:, 1:].values  # 去掉第一列
Y_train_df = pd.read_excel(r"D:\GitHub\modeling\D\ans3\ADMET.xlsx", engine='openpyxl', sheet_name='training')
Y = Y_train_df.iloc[:, 1].values  # 一个label

# 2 分割训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

# 3 在训练之前，我们需要对数据进行规范化，这样让数据同在同一个量级上，避免因为维度问题造成数据误差：
# 采用 Z-Score 规范化数据，保证每个特征维度的数据均值为 0，方差为 1
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
test_X = ss.transform(x_test)

# 4 创建 SVM 分类器
model = svm.SVC(kernel='rbf')
# 用训练集做训练
model.fit(x_train, y_train)
# 用测试集做预测
prediction = model.predict(test_X)
print('准确率: ', metrics.accuracy_score(prediction, y_test))

# xgboost
xgb2 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=7,
    min_child_weight=1,
    gamma=0,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
modelfit(xgb2, x_train, y_train, x_test, y_test)

# 6 随机森林
# model_rf = RandomForestClassifier(n_estimators = 1000, max_depth=None, random_state=0)
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)
# 用测试集做预测
p_rf = model_rf.predict(x_test)
print('随机森林准确率: ', metrics.accuracy_score(p_rf, y_test))
