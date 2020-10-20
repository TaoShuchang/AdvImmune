import numpy as np
import cvxpy as cp
import scipy.sparse as sp
from scipy.sparse.linalg import gmres
import time

import warnings
from joblib import Parallel, delayed
from utils import flip_edges, top_k_numba, edges_to_sparse
import gurobipy

def policy_iteration(adj, alpha, fragile, local_budget, reward, teleport, max_iter=1000):
    """
    Performs policy iteration to find the set of fragile edges to flip that maximize (r^T pi),
    where pi is the personalized PageRank of the perturbed graph.
    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    alpha : float
        (1-alpha) teleport[v] is the probability to teleport to node v.
    fragile : np.ndarray, shape [?, 2]
        Fragile edges that are under our control.
    local_budget : np.ndarray, shape [n]
        Maximum number of local flips per node.
    reward : np.ndarray, shape [n]
        Reward vector.
    teleport : np.ndarray, shape [n]
        Teleport vector.
        Only used to compute the objective value. Not needed for optimization.
    max_iter : int
        Maximum number of policy iterations.
    Returns
    -------
    opt_fragile : np.ndarray, shape [?, 2]
        Optimal fragile edges.
    obj_value : float
        Optimal objective value.
    adj_changing: sp.spmatrix, shape [n, n]
    """
    n = adj.shape[0]

    cur_fragile = np.array([])
    cur_obj_value = np.inf
    prev_fragile = np.array([[0, 0]])
    max_obj_value = -np.inf

    # if the budget is a scalar set the same budget for all nodes
    if not isinstance(local_budget, np.ndarray):
        local_budget = np.repeat(local_budget, n)

    # does standard value iteration
    for it in range(max_iter):
        adj_flipped = flip_edges(adj, cur_fragile)

        # compute the mean reward before teleportation
        trans_flipped = sp.diags(1 / adj_flipped.sum(1).A1) @ adj_flipped
        # trans_flipped = sp.diags(1 / adj_flipped.sum(1)) @ adj_flipped
        mean_reward = gmres(sp.eye(n) - alpha * trans_flipped, reward)[0]

        # compute the change in the mean reward
        vi = mean_reward[fragile[:, 0]]
        vj = mean_reward[fragile[:, 1]]
        ri = reward[fragile[:, 0]]
        change = vj - ((vi - ri) / alpha)

        # +1 if we are adding a node, -1 if we are removing a node
        add_rem_multiplier = 1 - 2 * adj[fragile[:, 0], fragile[:, 1]].A1
        # add_rem_multiplier = 1 - 2 * adj[fragile[:, 0], fragile[:, 1]]
        change = change * add_rem_multiplier

        # only consider the ones that improve our objective function
        improve = change > 0
        frag = edges_to_sparse(fragile[improve], n, change[improve])
        # select the top_k fragile edges
        cur_fragile = top_k_numba(frag, local_budget)

        # compute the objective value
        cur_obj_value = mean_reward @ teleport * (1 - alpha)

        # check for convergence
        edges_are_same = (edges_to_sparse(prev_fragile, n) - edges_to_sparse(cur_fragile, n)).nnz == 0
        if edges_are_same or np.isclose(max_obj_value, cur_obj_value):
            break
        else:
            prev_fragile = cur_fragile.copy()
            max_obj_value = cur_obj_value

    del trans_flipped, mean_reward, frag

    return cur_fragile, cur_obj_value
