import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers import GraphAttentionLayer


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, num_nodes, feat_dim):
        super(CNNEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 10, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 2, kernel_size=1, stride=1, padding=0)
        )
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim

    def forward(self, input):
        """

        :param input: [batch, time, num_nodes, feat_dim][128, 10, 16, 16]
        :return: [batch, num_nodes, num_nodes]
        """
        x = self.cnn(input)  # [128, 2, 16, 16]
        x = x.permute((0, 2, 3, 1))
        return x


class CNN_FullEncoder(nn.Module):
    def __init__(self, in_channels, num_nodes, feat_dim):
        super(CNN_FullEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=num_nodes, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=num_nodes, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=num_nodes, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 10, kernel_size=num_nodes, stride=1, padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 2, kernel_size=num_nodes, stride=1, padding=0)
        )
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim

    def forward(self, input):
        """

        :param input: [batch, time, num_nodes, feat_dim][128, 10, 16, 16]
        :return: [batch, num_nodes, num_nodes]
        """
        x = self.cnn(input)  # [128, 2, 16, 16]
        x = x.permute((0, 2, 3, 1))
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        # self.MLP = nn.Sequential(
        #     nn.Linear(nhid * nheads, nfeat * 2),
        #     nn.ReLU(),
        #     nn.Linear(nfeat * 2, nfeat)
        # )

    def forward(self, x, adj, pred_step=1):
        result = torch.zeros_like(x)  # x[128, 10, 16, 16]
        for i in range(x.size(1) - pred_step):
            pred = x[:, i, :, :]
            pred = F.dropout(pred, self.dropout, training=self.training)
            pred = torch.cat([att(pred, adj) for att in self.attentions], dim=2)  # output pred [128, 16, 64]
            pred = F.dropout(pred, self.dropout, training=self.training)
            pred = F.relu(self.out_att(pred, adj))  # output pred [128, 16, 16]
            # pred = F.log_softmax(pred, dim=1)
            result[:, i, :, :] += pred
        pred_all = result[:, :-pred_step, :, :].contiguous()
        return pred_all

