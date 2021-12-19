import numpy as np
import torch
import math
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score


def load_data(args):
    """
    :param args:
    :return: data and network
    cascde [num_batch, num_timesteps, num_nodes, num_dims]
    network [num_batch, num_nodes, num_nodes]
    """
    X_train = np.load('data/' + args.load_data_log + '_train_cascade.npy')
    edges_train = np.load('data/' + args.load_data_log + '_train_network.npy')

    X_valid = np.load('data/' + args.load_data_log + '_valid_cascade.npy')
    edges_valid = np.load('data/' + args.load_data_log + '_valid_network.npy')

    X_test = np.load('data/' + args.load_data_log + '_test_cascade.npy')
    edges_test = np.load('data/' + args.load_data_log + '_test_network.npy')

    X_train = X_train.transpose((0, 2, 1, 3))
    X_valid = X_valid.transpose((0, 2, 1, 3))
    X_test = X_test.transpose((0, 2, 1, 3))

    X_train = torch.tensor(X_train)
    edges_train = torch.LongTensor(edges_train)

    X_valid = torch.tensor(X_valid)
    edges_valid = torch.LongTensor(edges_valid)

    X_test = torch.tensor(X_test)
    edges_test = torch.LongTensor(edges_test)

    train_data = TensorDataset(X_train, edges_train)
    valid_data = TensorDataset(X_valid, edges_valid)
    test_data = TensorDataset(X_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def nll_gaussian_loss(preds, target):
    neg_log_p = 0.5 * (preds - target) ** 2
    return neg_log_p.sum() / target.size(0)


def softargmax(pred, index, beta=1000):  # pred[128, 16, 16, 2]
    x = pred * beta
    x = x - torch.max(x, dim=3, keepdim=True).values
    x_exp = torch.exp(x)
    partition = x_exp.sum(dim=3, keepdim=True)
    result = torch.sum(x_exp / partition * index, dim=3)
    return result


def edge_accuracy_f1(preds, targets):
    # _, preds = preds.max(-1)  # 返回input张量中所有元素的最大值, dim=-1并减少一个维度
    acc, pre, rec, f1 = [], [], [], []
    pred = preds.cpu().detach().numpy().reshape((preds.size(0), -1))
    target = targets.cpu().detach().numpy().reshape((targets.size(0), -1))
    for i, j in zip(target, pred):
        # if np.sum(j) == 0:
        #     print(1)
        acc.append(accuracy_score(i, j))
        pre.append(precision_score(i, j))
        rec.append(recall_score(i, j))
        f1.append(f1_score(i, j))
    return np.mean(acc), np.mean(f1), np.mean(pre), np.mean(rec)
