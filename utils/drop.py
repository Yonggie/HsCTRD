import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_sparse import coalesce
from utils.num_nodes import maybe_num_nodes
# from num_nodes import maybe_num_nodes


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(edge_index, edge_attr=None, p=0.5, force_undirected=False,
                num_nodes=None, training=True):
    r"""Randomly drops edges from the adjacency matrix
    :obj:`(edge_index, edge_attr)` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)
    """

    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    if not training:
        return edge_index, edge_attr

    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)

    mask = edge_index.new_full((row.size(0), ), 1 - p, dtype=torch.float)
    mask = torch.bernoulli(mask).to(torch.bool)

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


def drop_feature(x, drop_prob=0.5):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def dropout_edge_with_p(edge_index, p_mask, edge_attr=None, force_undirected=False, num_nodes=None, training=True):
    r"""Randomly drops edges
    :obj:`(edge_index, edge_attr)` with probability mask.

    Args:
        edge_index (LongTensor): The edge indices.
        p_mask (Tensor, optional): Dropout probability mask matrix.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)
    """
    p_mask=probablize_mask(p_mask,'linear')
    if sum(sum(p_mask>1))>0 or sum(sum(p_mask<0))>0 :
        raise ValueError('Dropout probability has to be between 0 and 1')

    if not training:
        return edge_index, edge_attr

    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)

    # mask = edge_index.new_full((row.size(0), ), 1 - p, dtype=torch.float)
    mask = torch.bernoulli(p_mask).to(torch.bool)

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr

def drop_feature_with_p(x, p_mask):
    p_mask = probablize_mask(p_mask, 'linear')
    mask = torch.bernoulli(p_mask).to(torch.bool)
    # drop_mask = torch.empty(
    #     (x.size(1), ),
    #     dtype=torch.float32,
    #     device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    # x[:, mask] = 0
    x[mask] = 0

    return x

def convert(p,k=None,b=None):
    '''
    convert non-zero-one to zero-one with linear-clip function
    :param p: original value
    :param k: k in [0, +∞]
    :param b: b in [0,1], 0 if normal drop, 1 preserve all edges.
    :return: converted probability
    '''

    # y=k*p+b
    y=0.1*p+0
    y[y>1]=1
    y[y<0]=0
    return y

def probablize_mask(p_mask, convert_func):
    if convert_func=='linear':
        func=convert
    elif convert_func=='sigmoid':
        func=nn.Sigmoid()
    elif convert_func=='tanh':
        func = nn.Tanh()
    tmp=p_mask.clone()
    p_mask=func(tmp.detach())

    return p_mask

def shuffle_corrupt(features):
    # shuffled features (corruption operation)
    nb_nodes=len(features)
    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[idx]
    return shuf_fts

