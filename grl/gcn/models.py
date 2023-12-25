import torch.nn as nn
import torch.nn.functional as F
from grl.gcn.layers import GraphConvolution


# GCN模型 （输入特征维度，隐藏层维度，输出层维度，dropout）
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)  # 第一层GCN
        self.gc2 = GraphConvolution(nhid, nclass)  # 第二层GCN
        self.dropout = dropout  # 输出层

    # 前向传播（输入特征 和 邻接矩阵）
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # 第一层GCN 过ReLU
        x = F.dropout(x, self.dropout, training=self.training)  # 随机Dropout特征
        x = self.gc2(x, adj)  # 第二层GCN
        return F.log_softmax(x, dim=1)  # 第二层过log_softmax函数
