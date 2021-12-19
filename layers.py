import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpah = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpah)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # self.a [16, 1], self.Wh1 [128, 16, 1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + torch.transpose(Wh2, dim0=1, dim1=2)  # e [128, 16, 16]
        return self.leakyrelu(e)

    def forward(self, h, adj):  # h[128, 16, 16]  adj[128. 16, 16]
        # adj 是邻接矩阵，h是[num_nodes, feature]矩阵
        Wh = torch.matmul(h, self.W)  # self.W [16, 8]  Wh[128, 16, 8]
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        torch_0 = torch.zeros_like(e)
        add_zero_vec = torch.where(adj > 0.5, torch_0, zero_vec)
        attention = e * adj + add_zero_vec
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)  # attention [128, 16, 16]
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
