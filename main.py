import numpy as np
import pandas as pd
import torch
import time
import argparse
import os

from utils import load_dataset, standardize, get_fragile
from AdvImmune import pagerank_adj_changing, grad_adv_immune, worst_margins_given_k_squared
torch.cuda.empty_cache()
np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def main(opts):
    dataset = opts['dataset']
    scen = opts['scenario']
    alpha = opts['alpha']
    att_local = opts['attackLocal']
    con_local = opts['immuneLocal']
    file_name = 'data/' + dataset + '.npz'
    output_file = 'output/' + dataset + '/' + scen
    # Load data
    graph = load_dataset(file_name=file_name)
    ori_adj, ori_attr = standardize(adj_matrix=graph['adj_matrix'], attr_matrix=graph['attr_matrix'])
    adj = torch.from_numpy(ori_adj.todense()).float()
    logits = np.load('data/' + dataset + '_logits.npy')
    labels = np.load('data/'+ dataset + '_labels.npy')

    deg = ori_adj.sum(1).A1.astype(np.int32)
    local_budget = np.maximum(deg-att_local, 0)
    con_budget_local = np.maximum(deg-con_local, 0)
    fragile = get_fragile(adj=ori_adj, threat_model=scen)
    n, nc = logits.shape
    edge_num = adj.sum().item()
    nodepair_num = n*n
    if scen == 'rem':
        # Remove-only
        num = edge_num
        start, end, interval = 0.005, 0.05, 0.005
    else:
        # Remove-Add
        num = nodepair_num
        start, end, interval = 0.001, 0.01, 0.001 
    # Initilization
    cur_controlled = torch.ones((n,n))
    # Obtain worst-case graph and its personalized PageRank
    ori_ppr_changing = pagerank_adj_changing(ori_adj, alpha, fragile, cur_controlled, local_budget, logits)
    np.save(output_file + '_ori_ppr_changing.npy', ori_ppr_changing)
    con_ppr_changing = ori_ppr_changing
    exceed_local = []
    # Compute immune graph matrix iteratively
    for con_ratio in np.arange(start, end, interval):
        if con_ratio > start:
            cur_control = np.aload(output_file + '_adj_controlled_%.1f%%.npy'%((con_ratio-interval)*100))
            cur_controlled = torch.Tensor(cur_control)
        cur_con_num = int(round(num * (con_ratio-interval)))
        con_budget = int(round(num * con_ratio))
        print('After immunizing {:.1f}% ({:d}) edges:'.format(con_ratio*100, con_budget))
        # Compute the nodes which exceed immune local budget, and 
        for exceed_i in range(n):
            cur_localcon = np.where(cur_controlled[exceed_i]==0)[0].shape[0]
            if cur_localcon == con_budget_local[exceed_i]:
                exceed_local.append(exceed_i)
        # Obtain immune graph matrix
        adj_controlled = grad_adv_immune(adj, con_ppr_changing, cur_controlled, exceed_local, cur_con_num, con_budget, logits, labels, alpha, con_budget_local)
        # Update worst-case graph and its personalized PageRank
        con_ppr_changing = pagerank_adj_changing(ori_adj, alpha, fragile, adj_controlled, local_budget, logits)
        con_worst_margins = worst_margins_given_k_squared(con_ppr_changing, labels, logits)
        print('After immunizing {:d}: Ratio of certified all nodes: {:.6f}'.format(con_budget, (con_worst_margins>0).mean()))
        print('No. of robust node:', (con_worst_margins>0).sum())
        print('Average of worst-case margin:', con_worst_margins.mean())

        adj_controlled = adj_controlled.detach().numpy()
        np.save(output_file + '_adj_controlled_%.1f%%.npy'%(con_ratio*100), adj_controlled)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AdvImmune')

    # configure
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--scenario', choices=['rem','add_rem'], default='rem', help='Scenarios of surrogate attack model')
    parser.add_argument('--alpha', default=0.85,help='alpha in personalized pagerank')

    # dataset
    parser.add_argument('--dataset', choices=['citeseer','cora','reddit'], default='citeseer',
                        help='dataset to use')
    
    # budgets
    parser.add_argument('--attackLocal', type=int, default=0, help='the local budget of surrogate model and robustness certification')
    parser.add_argument('--immuneLocal', type=int, default=0, help='the local budget of immunization')
#     parser.add_argument('--immuneGlobal', default=0.05, help='the global budget of immunization (ratio)')
    
    args = parser.parse_args()
    opts = args.__dict__.copy()
    att_sucess = main(opts) 

        