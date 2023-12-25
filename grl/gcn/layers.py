import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# 单层GCN层
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    # 输入维度，输出维度，偏置
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features  # 输入维度
        self.out_features = out_features  # 输出维度
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # 权重参数
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  # 偏置参数
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 参数重置

    # 参数重置
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向传播（输入层特征，邻接矩阵）
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # 输入层特征 * 权重  即 HW
        output = torch.spmm(adj, support)  # 稀疏矩阵乘法 即 AHW
        if self.bias is not None:
            return output + self.bias   # 偏置项
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
