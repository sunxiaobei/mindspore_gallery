import numpy as np
import scipy.sparse as sp
# coo_matrix是可以根据行和列索引进行data值的累加
# >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
# >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
# >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
# >>> sp.coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
# array([[3, 0, 1, 0],
#        [0, 2, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 1]])


# 拉普拉斯归一化 'NormLap': normalized_laplacian,  # A' = I - D^-1/2 * A * D^-1/2
def normalized_laplacian(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()  # 返回一个一维数组
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (sp.eye(adj.shape[0]) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()  # 返回稀疏矩阵的coo_matrix形式


# 拉普拉斯矩阵 'Lap': laplacian,  # A' = D - A
def laplacian(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1)).flatten()
   d_mat = sp.diags(row_sum)
   return (d_mat - adj).tocoo()


# 自环归一化GCN  'FirstOrderGCN': gcn,  # A' = I + D^-1/2 * A * D^-1/2  2016 GCN
def gcn(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (sp.eye(adj.shape[0]) + d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


# 邻接矩阵 + （D+I）归一化 + 自环  'BingGeNormAdj': bingge_norm_adjacency,  # A' = I + (D + I)^-1/2 * (A + I) * (D + I)^-1/2
def bingge_norm_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) + sp.eye(adj.shape[0])).tocoo()


#  邻接矩阵 + （D+I）归一化  'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


# 自环 + 归一化（邻接矩阵 + 随机游走 + 拉普拉斯） 'RWalkLap': random_walk_laplacian,  # A' = I - D^-1 * A
def random_walk_laplacian(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return (sp.eye(adj.shape[0]) - d_mat.dot(adj)).tocoo()


# 邻接矩阵adj对称归一化并返回coo存储模式  不加自环
# 对称归一化（邻接矩阵 + 度归一化）  'NormAdj': normalized_adjacency,  # D^-1/2 * A * D^-1/2
def normalized_adjacency(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)  # adj 邻接矩阵（稀疏化）
    row_sum = np.array(adj.sum(1))  # row_sum 按行求和  度
    row_sum = (row_sum == 0) * 1 + row_sum  # 避免求和为0  除数不能为0
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()  # D^{-1/2}  后展平
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # INF 转换成0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^{-1/2} 转换成对角矩阵
    # adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # 源GCN
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()  # 归一化  D^{-1/2} (A+I) D^{-1/2}  返回稀疏矩阵


# 自环归一化（邻接矩阵 + 随机游走 + 自环) 'AugRWalk': aug_random_walk,  # A' = (D + I)^-1*(A + I)
def aug_random_walk(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return (d_mat.dot(adj)).tocoo()


# 归一化（邻接矩阵 + 随机游走 ）'RWalk': random_walk,  # A' = D^-1*A   ，Random walk normalized Laplacian
def random_walk(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return d_mat.dot(adj).tocoo()


# 邻接矩阵 + 自环   'INorm': i_norm,  # A' = A + I
def i_norm(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    return adj


# 邻接矩阵  'NoNorm': no_norm,  # A' = A
def no_norm(adj):
   adj = sp.coo_matrix(adj)
   return adj


# 获取 normalization 方法
def fetch_normalization(type):
   switcher = {
       'NormLap': normalized_laplacian,  # A' = I - D^-1/2 * A * D^-1/2
       'Lap': laplacian,  # A' = D - A
       'RWalkLap': random_walk_laplacian,  # A' = I - D^-1 * A
       'FirstOrderGCN': gcn,   # A' = I + D^-1/2 * A * D^-1/2
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
       'BingGeNormAdj': bingge_norm_adjacency,  # A' = I + (D + I)^-1/2 * (A + I) * (D + I)^-1/2
       'NormAdj': normalized_adjacency,  # D^-1/2 * A * D^-1/2
       'RWalk': random_walk,  # A' = D^-1*A
       'AugRWalk': aug_random_walk,  # A' = (D + I)^-1*(A + I)
       'NoNorm': no_norm,  # A' = A
       'INorm': i_norm,  # A' = A + I
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func


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

