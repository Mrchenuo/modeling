import torch
import torch.nn as nn
from base import BaseModel


class Net(BaseModel):
    def __init__(self):
        super(Net2, self).__init__()

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 16, 3),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(16, 16, 3),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(16, 32, 3),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(4, 4),
        #     nn.Conv2d(32, 32, 3),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(4, 4),
        #     nn.Flatten(),
        # )
        self.mlp = nn.Sequential(
            # nn.Linear(288, 32),
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Softmax(1),
        )
        for param in self.mlp.parameters():
            nn.init.normal_(param, mean=0, std=0.01)

    def forward(self, x):
        '''
        params:
            x: [B, W, H, C]
        '''
        B, W = x.shape

        # x = x.transpose(1, 3)
        # t = self.cnn(x)
        y_hat = self.mlp(x)
        return y_hat


class Net2(BaseModel):

    def __init__(self, features=20):
        super(Net2, self).__init__()

        self.linear_relu1 = nn.Linear(features, 128)
        # self.bn_1 = nn.BatchNorm1d(128)
        self.linear_relu2 = nn.Linear(128, 256)
        # self.bn_2 = nn.BatchNorm1d(256)
        self.linear_relu3 = nn.Linear(256, 256)
        # self.bn_3 = nn.BatchNorm1d(256)
        self.linear_relu4 = nn.Linear(256, 128)
        # self.bn_4 = nn.BatchNorm1d(128)
        self.linear5 = nn.Linear(128, 1)

    def forward(self, x):
        y_pred = self.linear_relu1(x)
        # y_pred = self.bn_1(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu2(y_pred)
        # y_pred = self.bn_2(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu3(y_pred)
        # y_pred = self.bn_3(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu4(y_pred)
        # y_pred = self.bn_4(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear5(y_pred)
        return y_pred


class TabularModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(20)
        self.lin1 = nn.Linear(20, 500)
        # relu
        self.bn1 = nn.BatchNorm1d(500)
        self.lin2 = nn.Linear(500, 100)
        # relu
        self.bn2 = nn.BatchNorm1d(100)
        self.lin3 = nn.Linear(100, 1)

    def forward(self, x_in):
        # print(x_in.shape)
        x = self.bn_in(x_in)
        x = nn.functional.relu(self.lin1(x))
        x = self.bn1(x)
        # print(x)

        x = nn.functional.relu(self.lin2(x))
        x = self.bn2(x)
        # print(x)

        x = self.lin3(x)
        # x = torch.sigmoid(x)
        return x
