import os

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn import preprocessing
from torch.utils.data import ConcatDataset
from torch.utils.data.dataset import TensorDataset
from torchvision import datasets, transforms
from base import BaseDataLoader


class RIDataLoader(BaseDataLoader):
    def __init__(self, test_X, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        # 加载数据
        var_top20 = [659, 403, 587, 476, 531, 56, 357, 39, 410, 351, 673, 639, 79, 665, 154, 716, 23, 237, 661, 727]
        # var_top20 = [660, 404, 588, 477, 532, 57,  358, 40, 411, 352, 674, 640, 80, 666, 155, 717, 24, 238, 662, 728]
        test_X = test_X[:, var_top20]

        Y = test_X[:, 1]
        # data_dir = r"D:\GitHub\modeling\D\data\ERα_activity.xlsx"
        # train_df = pd.read_excel(data_dir, engine='openpyxl', sheet_name='training')
        # train_df = train_df.iloc[:, 2]  # 最后两行为nan去掉，第一列为名称去掉
        # Y = train_df.values

        data_X = preprocessing.MinMaxScaler().fit_transform(test_X)
        train_features = torch.tensor(data_X, dtype=torch.float32)
        train_labels = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)
        self.dataset = TensorDataset(train_features, train_labels)

        # data_dir = r""D:\GitHub\modeling\D\ans3\Molecular_Descriptor.xlsx""
        # train_df = pd.read_excel(data_dir, engine='openpyxl', sheet_name='training')
        # train_df = train_df.iloc[:, 1:]  # 最后两行为nan去掉，第一列为名称去掉
        # Y = train_df.iloc[:, -1].values
        # var_top20 = np.load(r'D:\GitHub\modeling\D\feature_extraction/variable_top20.npy')
        # # var_top20 = pd.Index(var_top20)
        # var = ['MDEC-23', 'minsssN', 'LipoaffinityIndex', 'maxHsOH', 'maxssO', 'C1SP2',
        #        'minHsOH', 'BCUTc-1l', 'minsOH', 'minHBint5', 'MLFER_A', 'nHBAcc', 'VC-5',
        #        'MDEO-12', 'ndssC', 'TopoPSA', 'ATSc3', 'SHBint10', 'MDEC-33', 'XLogP']
        # var_top20 = pd.Index(var)
        # train_df = train_df[var_top20]
        # X = train_df.values
        #
        # # 将每个特征值归一化到一个固定范围
        # # 原始数据标准化，为了加速收敛
        # # 最小最大规范化对原始数据进行线性变换，变换到[0,1]区间
        # data_X = preprocessing.MinMaxScaler().fit_transform(X)
        # train_features = torch.tensor(data_X, dtype=torch.float32)
        # train_labels = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)
        # self.dataset = TensorDataset(train_features, train_labels)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
