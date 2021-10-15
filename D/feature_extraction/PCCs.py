import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # 从硬盘读取数据进入内存
    train_df = pd.read_excel("../data/train.xlsx", engine='openpyxl', sheet_name='Sheet2')
    train_df = train_df.iloc[:-2, 1:]  # 最后两行为nan去掉，第一列为名称去掉
    # train_df = train_df.iloc[:, 1:]  # 最后两行为nan去掉，第一列为名称去掉

    # # 根据皮尔逊相关系数选择与要预测的属性列SalePrice相关性最高的10个属性
    # # [:11]，选出11个是因为SalePrice自己与自己的相关性最高，所以要将它去除故选择排序后的前11个属性，再去除SalePrice
    # features = train_df.corr()['pIC50'].abs().sort_values(ascending=False)
    # features.drop('pIC50', axis=0, inplace=True)
    # features = features.index
    # print(features[:30])

    # 先用皮尔逊系数粗略的选择出相关性系数的绝对值大于0.3的属性列，这样不需要训练过多不重要的属性列
    # 可以这么做而且不会破坏实验的控制变量原则，因为根据皮尔逊相关系数选择出的重要性排名前10的属性列
    # 它们与要预测的属性列的皮尔逊相关系数均大于0.3，可以当成步骤1中也进行了同样的取相关系数为0.3的操作
    features = train_df.corr().columns[train_df.corr()['pIC50'].abs() > .3]
    features = features.drop('pIC50')

    # 使用随机森林模型进行拟合的过程
    X_train = train_df[features]
    y_train = train_df['pIC50']
    feat_labels = X_train.columns

    rf = RandomForestRegressor(n_estimators=1000, max_depth=None)
    rf_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('standardize', StandardScaler()), ('rf', rf)])
    rf_pipe.fit(X_train, y_train)

    # 根据随机森林模型的拟合结果选择特征
    rf = rf_pipe.__getitem__('rf')
    importance = rf.feature_importances_

    # np.argsort()返回待排序集合从下到大的索引值，[::-1]实现倒序，即最终imp_result内保存的是从大到小的索引值
    imp_result = np.argsort(importance)[::-1][:20]

    # 按重要性从高到低输出属性列名和其重要性
    for i in range(len(imp_result)):
        print("%2d. %-*s %f" % (i + 1, 30, feat_labels[imp_result[i]], importance[imp_result[i]]))

    # 对属性列，按属性重要性从高到低进行排序
    feat_labels = [feat_labels[i] for i in imp_result]
    a = np.array(feat_labels)
    np.save('variable_top20', a)  # 保存为.npy格式
    # 绘制特征重要性图像
    plt.title('Feature Importance')
    plt.bar(range(len(imp_result)), importance[imp_result], color='lightblue', align='center')
    plt.xticks(range(len(imp_result)), feat_labels, rotation=90)
    plt.xlim([-1, len(imp_result)])
    plt.tight_layout()
    plt.show()