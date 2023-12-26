import numpy as np
import scipy.sparse as sp
# import torch
import mindspore as ms
from mindspore import ops
import sys
# 设置最大递归次数
sys.setrecursionlimit(99999)
"""
工具类
"""

# 归一化  D^{-1/2}AD^{-1/2}
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))  # 度 D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^{-1/2}
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 异常值处理
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 对角
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^{-1/2}AD^{-1/2}  Coo稀疏矩阵

# 行归一化  特征归一化 n*d  和  邻接矩阵归一化 按行 n*n
# 归一化  特征矩阵  输入特征矩阵  preprocess_features  Lil格式稀疏矩阵
# 处理特征:特征矩阵进行归一化并返回一个格式为(coords, values, shape)的元组
# 特征矩阵的每一行的每个元素除以行和，处理后的每一行元素之和为1
# 处理特征矩阵，跟谱图卷积的理论有关，目的是要把周围节点的特征和自身节点的特征都捕捉到，同时避免不同节点间度的不均衡带来的问题
def row_normalize(mx):
    """Row-normalize sparse matrix"""
    # a.sum()是将矩阵中所有的元素进行求和;a.sum(axis = 0)是每一列列相加;a.sum(axis = 1)是每一行相加
    rowsum = np.array(mx.sum(1))  # 稀疏矩阵mx 转换成np 特征按行求和，得到一个列向量，2708*1
    rowsum = (rowsum == 0) * 1 + rowsum  # 如果和为0，则转换成1 ，rowsum是除数，不能为0
    r_inv = np.power(rowsum, -1).flatten()  # 计算rowsum^{-1} 并展平，即转换成一维 2708
    r_inv[np.isinf(r_inv)] = 0.  # INF补零
    r_mat_inv = sp.diags(r_inv)  # 特征和 转换成对角矩阵（稀疏）2708 * 2708
    mx = r_mat_inv.dot(mx)  # 归一化后的特征 = 特征和的倒数（对角阵 d*d维）* 特征矩阵（n*d）
    return mx

# 计算acc （输出结果，标签）
def accuracy(output, labels):
    # preds = output.max(1)[1].type_as(labels)  # 将预测结果转化成labels格式
    # correct = preds.eq(labels).double()  # 预测结果与标签一致
    # correct = correct.sum()  # 预测标签正确的数量
    pred = np.argmax(output.asnumpy(), axis=1)
    correct = (pred == labels.asnumpy()).sum()
    return correct / len(labels)  # 预测标签正确的数量/总的预测数量


# # 稀疏矩阵转换成Tensor稀疏矩阵  输入稀疏矩阵  2708*2708
# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 转换成稀疏矩阵tocoo 32位浮点型
#     indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # 稀疏矩阵下标
#     values = torch.from_numpy(sparse_mx.data)  # 矩阵值
#     shape = torch.Size(sparse_mx.shape)  # 矩阵维度 2708*2708
#     return torch.sparse.FloatTensor(indices, values, shape)  # Tensor版稀疏矩阵


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
    # classes = set(labels)  # 所有标签集合
    classes = sorted(list(set(labels)))  # 排序后，固定标签
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}  # 遍历所有标签去重之后
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)  # 转换成One-hot
    return labels_onehot


# 采样mask，数据集标记 train test val
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


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
    # adj = sp.coo_matrix(adj)  # 邻接矩阵稀疏化 Coo
    # adj = adj + adj.T.multiply(adj.T > adj) + sp.eye(nodes_num)  # 对称矩阵
    # 矩阵转换成对称矩阵
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    nor_adj = normalize_adj(adj)  # 对称归一化
    adj = np.array(nor_adj.todense())  # 稠密矩阵  目前MS 无法稀疏矩阵乘法
    # 加入自环 归一化邻接矩阵
    # adj = row_normalize(adj + sp.eye(adj.shape[0]))

    # 特征归一化
    features = row_normalize(features)

    # 半监督节点分类数据集划分
    idx_train = [i for i in range(140)]
    idx_val = [i for i in range(200, 500)]
    idx_test = [i for i in range(500, 1500)]

    features = ms.Tensor(np.array(features.todense()), ms.float32)
    # labels = ms.Tensor(np.where(labels)[1], ms.int32)
    labels = ms.Tensor(np.where(labels)[1], ms.int32)
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
