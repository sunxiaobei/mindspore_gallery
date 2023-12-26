from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
import mindspore as ms
from mindspore import nn
from mindspore import ops

from grl.gcn.utils import load_data, accuracy
from grl.gcn.models import GCN


# Training settings
parser = argparse.ArgumentParser()
# Cuda GPU启用
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
# 数据集路径  调试 ../data/   ssh data/
parser.add_argument('--path', default="data/grl/", help="The data path.")
# 数据集 cora
parser.add_argument('--dataset', default="cora", help="The data set")
# 加速模型，训练时不作验证
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

ms.set_context(device_target='GPU', device_id=0)
np.random.seed(args.seed)  # 设置np.random
# # args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)  # 设置cuda.random

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.path, args.dataset)


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,    # labels.max().item() + 1    labels.shape[1]
            dropout=args.dropout)
# model.add_flags_recursive(fp16=True)

# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)

# if args.cuda:
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()
adj = ms.Tensor(adj)
feature = ms.Tensor(features)
# idx_train = ms.Tensor(idx_train)
# idx_val = ms.Tensor(idx_val)
# idx_test = ms.Tensor(idx_test)

def train(epoch):
    t = time.time()
    # model.train()
    # optimizer.zero_grad()
    model.set_train()

    # Define forward function
    def forward_fn(data, label, adj):
        output = model(data, adj)
        loss = ops.nll_loss(output[idx_train], label[idx_train])
        acc = accuracy(output[idx_train], label[idx_train])
        return loss, acc, output

    # Get gradient function
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    (loss_train, acc_train, output), grads = grad_fn(features, labels, adj)
    optimizer(grads)

    # loss_train.backward()  # ms 自动求导
    # optimizer.step()  # ms 自动更新

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        # model.eval()
        model.set_train(False)
        output = model(features, adj)

    loss_val = ops.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.asnumpy()),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.asnumpy()),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    # model.eval()
    model.set_train(False)
    output = model(features, adj)
    loss_test = ops.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.asnumpy()),
          "accuracy= {:.4f}".format(acc_test))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
