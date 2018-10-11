# coding: utf-8
import numpy as np
import scipy.sparse as sp
path="../data/cora/"
dataset="cora"
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                    dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:,1:-1], dtype=np.float32)
get_ipython().magic('paste')
labels = encode_onehot(idx_features_labels[:,-1])
labels
labels.shape
features.shape
features[0,:]
dir(features[0,:])
features[0,:].data
features[0,:].data.shape
features[0,:].data[data>0]
d = features[0,:].data
d[d>0]
d[d>0].shape
dir(features[0,:])
d = features[0,:].indices
d
d.shape
features.shape
d.nonzero
d.nonzero()
d
d = features[0,:].data
d.nonzero()
d.nonzero().shape
d.nonzero()[0]
d.nonzero()[0].shape
features > features.T
labels
idx = np.array(idx_features_labels[:,0], dtype=np.int32)
idx.shape
idx
idx_map = {j:i for i, j in enumerate(idx)}
idx_map[0]
idx_map
inv_idx_map = {idx_map[x]:x for x in idx_map}
inv_idx_map[0]
idx[0]
labels
np.where(labels)
help(np.where)
help(np.where)
idx_map
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
edges.shape
edges_unordered.shape
edges_unordered
edges
map(idx_map.get, edges_unordered.flatten())
list(map(idx_map.get, edges_unordered.flatten()))
edges_unordered
edges
edges[0,0]
edges_unordered[0,0]
idx_map(edges_unordered[0,0])
idx_map.get(edges_unordered[0,0])
help(sp.coo_matrix)
idx_batch1 = range(70)
idx_batch1
[x for x in idx_batch1]
idx_batch2 = range(70,140)
[x for x in idx_batch2]
adj
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix(np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
adj
adj.nonzero
adj.nonzero()
adj[0,:]
adj + sp.eye(adj.shape[0]))
adj + sp.eye(adj.shape[0])
adj_csr = sp.csr_matrix(adj)
adj[idx_batch1,idx_batch1]
adj_csr[idx_batch1,idx_batch1]
idx_batch1
adj_csr[idx_batch1,:]
adjtemp = adj_csr[idx_batch1,:]
adjtemp = adjtemp[:, idx_batch1]
adjtemp
features
features[idx_batch1,:]
adj_csr
adj_csr[idx_batch1,idx_batch1]
adj_csr[:,idx_batch1]
adj_csr[:,idx_batch1]
adj_csr[idx_batch1, :]
adj_csr.shape
features
labels
get_ipython().magic('paste')
make_batch(adj_csr, features, labels, idx_batch1)
type(adj_csr)
get_ipython().magic('paste')
make_batch(adj_csr, features, labels, idx_batch1)
get_ipython().magic('paste')
make_batch(adj_csr, features, labels, idx_batch1)
make_batch(adj_csr, features, labels, idx_batch2)
res = make_batch(adj_csr, features, labels, idx_batch2)
len(res)
res[0]
res[1]
features.shape
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                    dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:,1:-1], dtype=np.float32)
features.shape
res[1]
res[2]
res[2].shape
res[3]
import torch
res = make_batch(adj_csr, features, labels, idx_batch2)
get_ipython().magic('paste')
res = make_batch(adj_csr, features, labels, idx_batch2)
get_ipython().magic('paste')
res = make_batch(adj_csr, features, labels, idx_batch2)
len(res)
[len(x) for x in res]
[x.shape for x in res if hasattr(x,'shape')]
[x.shape for x in res if hasattr(x,'shape') else x]
[x.shape for x in res if hasattr(x,'shape'): else x]
[x.shape if hasattr(x,'shape') else x for x in res]
res[-1]
labels.max().item()
labels.max()
labels
idx_features_labels[:,-1].max().item()
L = idx_features_labels[:,-1]
L
labels
import torch
L = torch.LongTensor(np.where(labels)[1]
)
L
L.max().item() + 1
get_ipython().magic('paste')
get_ipython().magic('paste')
class args:
    hidden = 16
    dropout = 0.5
    
args.hidden
args.dropout
get_ipython().magic('paste')
model
features.shape
adj.shape
model(features, adj)
get_ipython().magic('paste')
features.shape
type(features)
type(adj)
type(labels)
model(features, adj)
adj_, features_, labels_, idx_map_ = res
adj_.shape
features_.shape
labels_.shape
features_ = torch.FloatTensor(np.array(features_.todense()))
labels_ = torch.LongTensor(np.where(labels_)[1])
adj_ = sparse_mx_to_torch_sparse_tensor(adj_)
#output = '\n'.join([','.join(map(unicode, row)) for row in edges])
model(features_, adj_)
output = model(features_, adj_)
output.shape
output[idx_batch2]
idx_map_
idx
idxs
idxs = list(map(idx_map_.get, batch_idx2))
batch_idx1
idxs = list(map(idx_map_.get, idx_batch2))
idxs
labels_
adj
get_ipython().magic('paste')
output = model(features_, adj_)
output.shape
labels_
import torch.nn.functional as F
F.nll_loss(output, labels_)
get_ipython().magic('pwd ')
get_ipython().magic('save playing_with_pygcn 1-185')
