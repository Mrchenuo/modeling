import torch
import sklearn.metrics as metrics
from torch import nn


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        tmp1 = torch.sum(pred == target)
        tmp2 = tmp1.item()
        correct += torch.sum(pred == target).item()
        tmp3 = correct / len(target)
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def f1_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        score = metrics.f1_score(target, pred, average='weighted')
    return score


def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        score = metrics.precision_score(target, pred, average='weighted')
    return score


def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        score = metrics.recall_score(target, pred, average='weighted')
    return score


def log_rmse(output, target):
    with torch.no_grad():
        loss = nn.MSELoss()
        # pred = torch.argmax(output, dim=1)
        # assert pred.shape[0] == len(target)
        # pred = pred.cpu().numpy()
        # target = target.cpu().numpy()
        # score = metrics.recall_score(target, pred, average='weighted')
        clipped_preds = torch.clamp(output, 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(target)))
    return rmse.item()
