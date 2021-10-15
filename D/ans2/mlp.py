import pickle
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
from d2l import torch as d2l
from torch.backends import cudnn

# 加载数据
train_df = pd.read_excel("../data/train.xlsx", engine='openpyxl', sheet_name='Sheet1')
train_df = train_df.iloc[:-2, 1:]  # 最后两行为nan去掉，第一列为名称去掉
Y = train_df.iloc[:, -1].values
var_top20 = np.load(r'../feature_extraction/variable_top20.npy')
var_top20 = pd.Index(var_top20)
train_df = train_df[var_top20]
X = train_df.values


# 将每个特征值归一化到一个固定范围
# 原始数据标准化，为了加速收敛
# 最小最大规范化对原始数据进行线性变换，变换到[0,1]区间
data_X = preprocessing.MinMaxScaler().fit_transform(X)

# 利用train_test_split 进行训练集和测试集进行分开
X_train, X_test, y_train, y_test = train_test_split(data_X, Y, test_size=0.3)
train_features = torch.tensor(X_train, dtype=torch.float32)
test_features = torch.tensor(X_test, dtype=torch.float32)
train_labels = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

# 训练
loss = nn.MSELoss()
in_features = train_features.shape[1]
num_hiddens1 = 128
num_hiddens2 = 64

def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(0.1),
        # nn.Linear(num_hiddens1, num_hiddens2),
        # nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(num_hiddens1, 1))
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train.to(device), y_train.to(device), X_valid.to(device), y_valid.to(device)


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net().to(device)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay,
                                   batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')

        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')

    #保存模型
    filename = "model_mlp_" + "_epochs_" + str(num_epochs) + "_lr_" +str(lr) + "_wd_" +str(weight_decay) + "_bs_" +str(batch_size) + "_acc_" + str(valid_l_sum / k) +"_"+ device.type +".pkl"
    f = open(filename, 'wb')
    pickle.dump(net, f)
    f.close()
    return train_l_sum / k, valid_l_sum / k

# 参数设置
k = 5
num_epochs = 100
lr = 0.1
weight_decay = 50
batch_size = 64
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True if use_cuda else False

train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
plt.show()
