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
from data_loader.data_set import RIDataset


class RIDataLoader(BaseDataLoader):
    """
    Recolored Image data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
                 phase="train"):
        self.data_dir = data_dir
        normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
        dataset_list = []
        if phase == 'train':
            # 加载数据
            train_df = pd.read_excel(data_dir, engine='openpyxl', sheet_name='Sheet1')
            train_df = train_df.iloc[:-2, 1:]  # 最后两行为nan去掉，第一列为名称去掉
            Y = train_df.iloc[:, -1].values
            var_top20 = np.load(r'D:\GitHub\modeling\D\feature_extraction/variable_top20.npy')
            var = ['MDEC-23', 'minsssN', 'LipoaffinityIndex', 'maxHsOH', 'maxssO', 'C1SP2',
                   'minHsOH', 'BCUTc-1l', 'minsOH', 'minHBint5', 'MLFER_A', 'nHBAcc', 'VC-5',
                   'MDEO-12', 'ndssC', 'TopoPSA', 'ATSc3', 'SHBint10', 'MDEC-33', 'XLogP']
            var_top20 = pd.Index(var)
            train_df = train_df[var_top20]
            X = train_df.values

            # 将每个特征值归一化到一个固定范围
            # 原始数据标准化，为了加速收敛
            # 最小最大规范化对原始数据进行线性变换，变换到[0,1]区间
            data_X = preprocessing.MinMaxScaler().fit_transform(X)
            train_features = torch.tensor(data_X, dtype=torch.float32)
            train_labels = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)
            self.dataset = TensorDataset(train_features, train_labels)

        elif phase == 'test':
            # 加载数据
            train_df = pd.read_excel(data_dir, engine='openpyxl', sheet_name='Sheet1')
            train_df = train_df.iloc[:-2, 1:]  # 最后两行为nan去掉，第一列为名称去掉
            Y = train_df.iloc[:, -1].values
            var_top20 = pd.np.load(r'D:\GitHub\modeling\D\feature_extraction/variable_top20.npy')
            var_top20 = pd.Index(var_top20)
            train_df = train_df[var_top20]
            X = train_df.values

            # 将每个特征值归一化到一个固定范围
            # 原始数据标准化，为了加速收敛
            # 最小最大规范化对原始数据进行线性变换，变换到[0,1]区间
            data_X = preprocessing.MinMaxScaler().fit_transform(X)
            train_features = torch.tensor(data_X, dtype=torch.float32)
            train_labels = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)
            self.dataset = TensorDataset(train_features, train_labels)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
