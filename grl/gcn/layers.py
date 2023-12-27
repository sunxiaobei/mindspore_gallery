import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter


# 单层GCN层
class GraphConvolution(nn.Cell):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    # 输入维度，输出维度，偏置
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features  # 输入维度
        self.out_features = out_features  # 输出维度
        # 初始化权重矩阵
        init_range = np.sqrt(6.0 / (self.out_features + self.in_features))
        initial = np.random.uniform(-init_range, init_range, (self.in_features, self.out_features)).astype(np.float32)
        self.weight = Parameter(Tensor(initial, ms.float32), name='w')  # 权重参数
        if bias:
            initial_bias = np.random.uniform(-init_range, init_range, (self.out_features, )).astype(np.float32)
            self.bias = Parameter(Tensor(initial_bias, ms.float32), name='b')  # 偏置参数
        else:
            self.bias = None

    # 前向传播（输入层特征，邻接矩阵）
    def construct(self, input, adj):
        support = ops.mm(input, self.weight)  # 输入层特征 * 权重  即
        # ops.SparseTensorDenseMatmul()    MatMul()  ops.matmul()
        output = ops.matmul(adj, support)  # 稀疏矩阵乘法 即 AHW
        if self.bias is not None:
            return output + self.bias   # 偏置项
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
