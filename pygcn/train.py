from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, training_split, make_batch
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

np.random.seed(187348292)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj, features, labels = load_data()
adj_csr = sp.csr_matrix(adj)
idxs = np.arange(adj.shape[0])
train_size = 500
val_size = 300
test_size = 1000
idx_train, idx_val, idx_test = training_split(idxs, train_size, val_size, test_size)


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

adj_train, features_train, labels_train, _ = make_batch(adj_csr, features, labels, idx_train)
adj_val, features_val, labels_val, _ = make_batch(adj_csr, features, labels, idx_val)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features_train, adj_train)
    print("Labels shape")
    print(labels.shape)
    print("Output shape")
    print(output.shape)
    semisupervised = np.arange(output.shape[0]/4)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train = F.nll_loss(output[semisupervised], labels_train[semisupervised])
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output, labels_train)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features_val, adj_val)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val = F.nll_loss(output, labels_val)
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output, labels_val)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


adj_test, features_test, labels_test, _ = make_batch(adj_csr, features, labels, idx_test)

def test():
    model.eval()
    output = model(features_test, adj_test)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test = F.nll_loss(output, labels_test)
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output, labels_test)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
