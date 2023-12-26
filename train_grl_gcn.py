from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
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
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
np.random.seed(args.seed)  # 设置np.random

ms.set_context(device_target='GPU', device_id=0)
ms.set_seed(args.seed)
# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.path, args.dataset)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,    # labels.max().item() + 1    labels.shape[1]
            dropout=args.dropout)
# model.add_flags_recursive(fp16=True)

optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)

adj = ms.Tensor(adj)
feature = ms.Tensor(features)


def train(epoch):
    t = time.time()
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

    if not args.fastmode:
        # Evaluate validation set performance separately
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
