import numba
import numpy as np
import scipy.sparse as sp
import torch
import random
from scipy.sparse.linalg import gmres

def correction_term(adj, opt_fragile, fragile):
    """
    Computes correction term needed to map x_v to ppr_v.
    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    opt_fragile : np.ndarray, shape [?, 2]
        Optimal fragile edges.
    fragile : np.ndarray, shape [?, 2]
        Fragile edges that are under our control.
    Returns
    -------
    correction : np.ndarray, shape [n]
        Correction term.
    """
    n = adj.shape[0]
    if len(opt_fragile) > 0:
        adj_all = adj + edges_to_sparse(fragile, n)
        adj_all[adj_all != 0] = 1
        deg_all = adj_all.sum(1).A1

        g_chosen = edges_to_sparse(opt_fragile, n, 1 - 2 * adj[opt_fragile[:, 0], opt_fragile[:, 1]].A1)
        n_removed = -g_chosen.multiply(g_chosen == -1).sum(1).A1
        n_added = g_chosen.multiply(g_chosen == 1).sum(1).A1
        n_to_add = edges_to_sparse(fragile, n, 1 - adj[fragile[:, 0], fragile[:, 1]].A1).sum(1).A1
        correction = 1 - (n_removed + (n_to_add - n_added)) / deg_all
    else:
        correction = np.ones(n)

    return correction

@numba.jit(nopython=True)
def _top_k(indices, indptr, data, k_per_row):
    """

    Parameters
    ----------
    indices: np.ndarray, shape [n_edges]
        Indices of a sparse matrix.
    indptr: np.ndarray, shape [n+1]
        Index pointers of a sparse matrix.
    data: np.ndarray, shape [n_edges]
        Data of a sparse matrix.
    k_per_row: np.ndarray, shape [n]
        Number of top_k elements for each row.
    Returns
    -------
    top_k_idx: list
        List of the indices of the top_k elements for each row.
    """
    n = len(indptr) - 1
    top_k_idx = []
    for i in range(n):
        cur_top_k = k_per_row[i]
        if cur_top_k > 0:
            cur_indices = indices[indptr[i]:indptr[i + 1]]
            cur_data = data[indptr[i]:indptr[i + 1]]
            # top_k = cur_indices[np.argpartition(cur_data, -cur_budget)[-cur_budget:]]
            top_k = cur_indices[cur_data.argsort()[-cur_top_k:]]
            top_k_idx.append(top_k)

    return top_k_idx


def top_k_numba(x, k_per_row):
    """
    Returns the indices of the top_k element per row for a sparse matrix.
    Considers only the non-zero entries.
    Parameters
    ----------
    x : sp.spmatrix, shape [n, n]
        Data matrix.
    k_per_row : np.ndarray, shape [n]
        Number of top_k elements for each row.
    Returns
    -------
    top_k_per_row : np.ndarray, shape [?, 2]
        The 2D indices of the top_k elements per row.
    """
    # make sure that k_per_row does not exceed the number of non-zero elements per row
    k_per_row = np.minimum(k_per_row, (x != 0).sum(1).A1)
    n = x.shape[0]
    row_idx = np.repeat(np.arange(n), k_per_row)

    col_idx = _top_k(x.indices, x.indptr, x.data, k_per_row)
    col_idx = np.concatenate(col_idx)

    top_k_per_row = np.column_stack((row_idx, col_idx))


    return top_k_per_row


def flip_edges(adj, edges):
    """
    Flip the edges in the graph (A_ij=1 becomes A_ij=0, and A_ij=0 becomes A_ij=1).

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    edges : np.ndarray, shape [?, 2]
        Edges to flip.
    Returns
    -------
    adj_flipped : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix with flipped edges.
    """
    adj_flipped = adj.copy().tolil()
#     if len(edges) > 0:
#         adj_flipped[edges[:, 0], edges[:, 1]] = 1 - adj[edges[:, 0], edges[:, 1]]
    if len(edges) > 0:
        for e in edges:
            adj_flipped[e[0], e[1]] = 1 - adj[e[0], e[1]]
    return adj_flipped


def compute_adj_changing(adj, opt_fragile):
    n = adj.shape[0]
    adj_changing = torch.zeros([n, n])

    for edge in opt_fragile:
        adj_changing[edge[0],edge[1]] = 1 - 2*adj[edge[0],edge[1]]
    del adj, opt_fragile
    return adj_changing


def propagation_matrix(adj, alpha=0.85, sigma=1):
    """
    Computes the propagation matrix  (1-alpha)(I - alpha D^{-sigma} A D^{sigma-1})^{-1}.

    Parameters
    ----------
    adj : tensor, shape [n, n]
    alpha : float
        (1-alpha) is the teleport probability.
    sigma
        Hyper-parameter controlling the propagation style.
        Set sigma=1 to obtain the PPR matrix.
    Returns
    -------
    prop_matrix : tensor, shape [n, n]
        Propagation matrix.
    """
    deg = adj.sum(1)
    deg_min_sig = torch.matrix_power(torch.diag(deg), -sigma)
    # 为了节省内存 100m
    if sigma - 1 == 0:
        deg_sig_min = torch.diag(torch.ones_like(deg))
    else:
        deg_sig_min = torch.matrix_power(torch.diag(deg), sigma - 1)

    n = adj.shape[0]
    pre_inv = torch.eye(n) - alpha * deg_min_sig @ adj @ deg_sig_min

    prop_matrix = (1 - alpha) * torch.inverse(pre_inv)
    del pre_inv,deg_min_sig, adj
    return prop_matrix


def topic_sensitive_pagerank(adj, alpha, teleport):
    """
    Computes the topic-sensitive PageRank vector.

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    alpha : float
        (1-alpha) teleport[v] is the probability to teleport to node v.
    teleport : np.ndarray, shape [n]
        Teleport vector.

    Returns
    -------
    ppr : np.ndarray, shape [n]
        PageRank vector.
    """
    assert np.isclose(teleport.sum(), 1)

    n = adj.shape[0]
    trans = sp.diags(1 / adj.sum(1).A1) @ adj.tocsr()

    # gets one row from the PPR matrix (since we transpose the transition matrix)
    ppr = sp.linalg.gmres(sp.eye(n) - alpha * trans.T, teleport)[0] * (1 - alpha)

    return ppr


def edges_to_sparse(edges, num_nodes, weights=None):
    """Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    :param edges: array-like, shape [num_edges, 2]
        Array with each row storing indices of an edge as (u, v).
    :param num_nodes: int
        Number of nodes in the resulting graph.
    :param weights: array_like, shape [num_edges], optional, default None
        Weights of the edges. If None, all edges weights are set to 1.
    :return: sp.csr_matrix
        Adjacency matrix in CSR format.
    """
    if weights is None:
        weights = np.ones(edges.shape[0])

    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes)).tocsr()


def get_fragile(adj, threat_model):
    """
    Generate a set of fragile edges corresponding to different threat models and scenarios.

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    threat_model : string
        'rem' specifies an attacker that can only remove edges, i.e. fragile edges are existing edges in the graph,
        'add_rem' specifies an attacker that can both add and remove edges.

    Returns
    -------
    fragile : np.ndarray, shape [?, 2]
        Set of fragile edges.
    """
    n = adj.shape[0]

    mst = sp.csgraph.minimum_spanning_tree(adj)
    mst = mst + mst.T

    if threat_model == 'rem':
        fragile = np.column_stack((adj - mst).nonzero())
    elif threat_model == 'add_rem':
        fragile_rem = np.column_stack((adj - mst).nonzero())
        fragile_add = np.column_stack(np.ones((n, n)).nonzero())
        fragile_add = fragile_add[adj[fragile_add[:, 0], fragile_add[:, 1]].A1 == 0]
        fragile_add = fragile_add[fragile_add[:, 0] != fragile_add[:, 1]]
        fragile = np.row_stack((fragile_add, fragile_rem))
    else:
        raise ValueError('threat_model not set correctly.')

    return fragile


def load_dataset(file_name):
    """
    Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """

    if not file_name.endswith('.npz'):
        file_name += '.npz'
    if file_name.endswith('reddit.npz')  or file_name.endswith('karate.npz'):
        with np.load(file_name, allow_pickle=True) as loader:
            loader = dict(loader)
            adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                        loader['adj_indptr']), shape=loader['adj_shape'])

            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                        loader['attr_indptr']), shape=loader['attr_shape'])

            labels = loader.get('labels')

            graph = {
                'adj_matrix': adj_matrix,
                'attr_matrix': attr_matrix,
                'labels': labels
            }
    else:
        with np.load(file_name, allow_pickle=True) as loader:
            loader = dict(loader)
            adj_matrix = sp.csr_matrix((loader['adj_matrix.data'], loader['adj_matrix.indices'],
                                        loader['adj_matrix.indptr']), shape=loader['adj_matrix.shape'])

            attr_matrix = sp.csr_matrix((loader['attr_matrix.data'], loader['attr_matrix.indices'],
                                        loader['attr_matrix.indptr']), shape=loader['attr_matrix.shape'])

            labels = loader.get('labels')

            graph = {
                'adj_matrix': adj_matrix,
                'attr_matrix': attr_matrix,
                'labels': labels
            }
    
    return graph


def standardize(adj_matrix, attr_matrix):
    """
    Make the graph undirected and select only the nodes belonging to the largest connected component.
    Parameters
    ----------
    adj_matrix : sp.spmatrix
        Sparse adjacency matrix
    attr_matrix : sp.spmatrix
        Sparse attribute matrix

    Returns
    -------
    standardized_adj_matrix: sp.spmatrix
        Standardized sparse adjacency matrix.
    standardized_attr_matrix: sp.spmatrix
        Standardized sparse attribute matrix.
    """
    # copy the input
    standardized_adj_matrix = adj_matrix.copy()

    # make the graph unweighted
    standardized_adj_matrix[standardized_adj_matrix != 0] = 1

    # make the graph undirected
    standardized_adj_matrix = standardized_adj_matrix.maximum(standardized_adj_matrix.T)

    # select the largest connected component
    _, components = sp.csgraph.connected_components(standardized_adj_matrix)
    c_ids, c_counts = np.unique(components, return_counts=True)
    id_max_component = c_ids[c_counts.argmax()]
    select = components == id_max_component

    standardized_adj_matrix = standardized_adj_matrix[select][:, select]
    standardized_attr_matrix = attr_matrix[select]

    # remove self-loops
    standardized_adj_matrix = standardized_adj_matrix.tolil()
    standardized_adj_matrix.setdiag(0)
    standardized_adj_matrix = standardized_adj_matrix.tocsr()
    standardized_adj_matrix.eliminate_zeros()

    return standardized_adj_matrix, standardized_attr_matrix

def unravel_index(index, array_shape):
    rows = index // array_shape[1]
    cols = index % array_shape[1]
    return rows, cols


def immune_edge_control(adj_controlled, sort_edge, con_local_budget, con_more_num):
    ori_num = np.where(adj_controlled==0)[0].shape[0]
    idx = np.array(sort_edge)[:,0]

    for i in idx:
        adj_controlled[i[0],i[1]] = 0
        if np.where(adj_controlled[i[0]] == 0)[0].shape[0] > con_local_budget[i[0]]:
            adj_controlled[i[0],i[1]] = 1

        cur_num = np.where(adj_controlled==0)[0].shape[0]
        if cur_num == ori_num+con_more_num:
            
            break

    return adj_controlled


def split(labels, n_per_class=20, seed=0):
    np.random.seed(seed)
    nc = labels.max() + 1

    split_train, split_val = [], []
    for l in range(nc):
        perm = np.random.permutation((labels == l).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])

    split_train = np.random.permutation(np.concatenate(split_train))
    split_val = np.random.permutation(np.concatenate(split_val))

    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

    split_test = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_train, split_val)))

    return split_train, split_val, split_test


def compute_grad_matrix(control_all, grad_all, con_budget_local, n):
    # 放入矩阵中 con_local=[[grad]] n*n矩阵每个元素都是grad
    con_local = np.zeros((n,n))
    for j in range(len(control_all)):
        control = control_all[j]
        grad = grad_all[j]
        for k in range(len(grad)):
            con_local[control[k]] = grad[k]
            
    # 满足控制的局部约束
    for j in range(n):
        con = con_local[j]
        con_budget_l = con_budget_local[j].astype('int32')
        if np.where(con!=0)[0].shape[0] > con_budget_l:
            more = (np.where(con!=0)[0].shape[0] - con_budget_l).astype('int32')
            idx_ord = np.argsort(con)[::-1]
            for k in range(con_budget_l, con_budget_l+more):
                con[idx_ord[k]] = 0
            con_local[j] = con
    
    return con_local

def compute_sort_edge(con_local):
    control_edges = {}
    row = np.where(con_local!=0)[0]
    col = np.where(con_local!=0)[1]
    control_edges={(row[i],col[i]): con_local[row[i],col[i]] for i in range(len(row))}
    sort_control_edges = sorted(control_edges.items(), key=lambda item:item[1], reverse=True)

    return sort_control_edges


def worstcase_class(ppr_flipped, labels, logits):
    n, nc = logits.shape
    worst_margins_all = np.ones((nc, nc, n)) * np.inf

    for c1 in range(nc):
        for c2 in range(nc):
            if c1 != c2:
                worst_margins_all[c1, c2] = (ppr_flipped[(c1,c2)].detach().numpy() @ (logits[:, c1] - logits[:, c2]))

    # selected the reference label according to the labels vector and find the minimum among all other classes
    worst_class = np.nanargmin(worst_margins_all[labels, :, np.arange(n)], 1)

    return worst_class

def compute_final_loss(loss, labels, worst_class):
    n = labels.shape[0]
    final_loss = torch.unsqueeze(loss[labels[0], worst_class[0]][0], 0)
    for i in range(1,n):
        tmp = torch.unsqueeze(loss[labels[i], worst_class[i]][i], 0)
        final_loss = torch.cat((final_loss, tmp), 0)

    return final_loss


def bisection(adj_controlled, a, b, perturbations, epsilon):
    def func(x):
        return torch.clamp(adj_controlled-x, 0, 1).sum() - perturbations

    miu = a
    while ((b-a) >= epsilon):
        miu = (a+b)/2
        # Check if middle point is root
        if (func(miu) == 0.0):
                break
        # Decide the side to repeat the steps
        if (func(miu)*func(a) < 0):
            b = miu
        else:
            a = miu
        # print("The value of root is : ","%.4f" % miu)
    return miu


def projection(adj_controlled, con_budget):
    # projected = torch.clamp(self.adj_controlled, 0, 1)
    if torch.clamp(adj_controlled, 0, 1).sum() > con_budget:
        left = (adj_controlled - 1).min()
        right = adj_controlled.max()
        miu = bisection(adj_controlled, left, right, con_budget, epsilon=1e-5)
        adj_controlled.data.copy_(torch.clamp(adj_controlled.data - miu, min=0, max=1))
    else:
        adj_controlled.data.copy_(torch.clamp(adj_controlled.data, min=0, max=1))
    return adj_controlled

def random_sample(adj, adj_controlled, ppr_adj_changing, logits, labels, con_budget, alpha):
    K = 100
    n,nc = logits.shape
    best_loss = -1000*torch.ones(n)
    with torch.no_grad():
        s = adj_controlled.detach().numpy()
        for i in range(K):
            sampled = np.random.binomial(1, s)
            ppr_flipped = {}
            loss = {}
            print(sampled.sum())
            if sampled.sum() > con_budget:
                continue
            adj_controlled.copy_(torch.Tensor(sampled))
            for c1 in range(nc):
                for c2 in range(nc):
                    if c1 != c2:
                        modified_adj = adj + torch.mul(ppr_adj_changing[(c1,c2)]['changing'], adj_controlled)
                        ppr_flipped[(c1,c2)] = propagation_matrix(adj=modified_adj, alpha=alpha)
                        tmp_re = torch.from_numpy(logits[:,c1] - logits[:,c2]).float()
                        loss[c1,c2] = ppr_flipped[(c1,c2)] @ tmp_re

            worst_class = worstcase_class(ppr_flipped, labels, logits)
            final_loss = compute_final_loss(loss, labels, worst_class)
            print(torch.sum(final_loss))
            if torch.sum(best_loss) < torch.sum(final_loss):
                best_loss = final_loss
                best_s = sampled
        adj_controlled.copy_(1 - torch.Tensor(best_s))

    return adj_controlled

# Fix seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True