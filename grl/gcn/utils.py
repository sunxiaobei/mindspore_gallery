import numpy as np
import scipy.sparse as sp
import torch
import sys
from grl.gcn.normalization import row_normalize
# 设置最大递归次数
sys.setrecursionlimit(99999)
"""
工具类
"""


# 计算acc （输出结果，标签）
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # 将预测结果转化成labels格式
    correct = preds.eq(labels).double()  # 预测结果与标签一致
    correct = correct.sum()  # 预测标签正确的数量
    return correct / len(labels)  # 预测标签正确的数量/总的预测数量


# 稀疏矩阵转换成Tensor稀疏矩阵  输入稀疏矩阵  2708*2708
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 转换成稀疏矩阵tocoo 32位浮点型
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # 稀疏矩阵下标
    values = torch.from_numpy(sparse_mx.data)  # 矩阵值
    shape = torch.Size(sparse_mx.shape)  # 矩阵维度 2708*2708
    return torch.sparse.FloatTensor(indices, values, shape)  # Tensor版稀疏矩阵


# 计算标准差  numpy.std() 求标准差的时候默认是除以 n 的，即是有偏的; numpy.std() 求标准差的时候默认是除以 n 的，即是有偏的
# np.std(a, ddof = 1)  ==  np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1)) ==  np.sqrt(( a.var() * a.size) / (a.size - 1))
# pandas.std() 默认是除以n-1 的，即是无偏的，如果想和numpy.std() 一样有偏，需要加上参数ddof=0 ，即pandas.std(ddof=0)
# https://www.zhangshengrong.com/p/l51g69oxX0/
def std(x, ddof=1):
    if len(x) == 1:
        return 0
    return np.std(x, ddof=1)


# One-hot编码  节点标签
def encode_onehot(labels):
    classes = set(labels)  # 所有标签集合
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}  # 遍历所有标签去重之后
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)  # 转换成One-hot
    return labels_onehot


# 读取数据集文件
def load_data_file(path="data/", dataset="cora", datatype="pygcn"):
    if datatype == "pygcn":
        path = path + "cora/"
    # 节点特征（ID + 特征 + 文字标签）
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # 获取特征矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 获取标签
    labels = encode_onehot(idx_features_labels[:, -1])  # 数据集最后一列是标签  需要处理

    # build graph 读取邻接表，转换成 邻接矩阵
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    return adj, features, labels

# https://arxiv.org/abs/1609.02907  GCN 节点对（源节点、目标节点） + 节点特征（ID + 特征 + 文字标签）
def load_data(path="data/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # 读取数据文件
    adj, features, labels = load_data_file(path=path, dataset="cora", datatype="pygcn")

    # 矩阵转换成对称矩阵
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 特征归一化
    features = row_normalize(features)

    # 加入自环 归一化邻接矩阵
    adj = row_normalize(adj + sp.eye(adj.shape[0]))

    # 半监督节点分类数据集划分
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
