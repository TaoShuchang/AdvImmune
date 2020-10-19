import numpy as np
import pandas as pd
import torch
import argparse
import os

from utils import load_dataset, standardize, get_fragile
from attack import *
from adv_immunity import *
torch.cuda.empty_cache()
np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

alpha = 0.85

def main(opts):
    
    dataset = opts['dataset']
    scen = opts['scenario']
    att_local = opts['attackLocal']
    con_local = opts['immuneLocal']
    con_global = opts['immuneGlobal']
    file_name = 'data/' + dataset + '.npz'
    output_file = 'output/' + dataset + '/' + scen
    # Load data
    graph = load_dataset(file_name=file_name)
    ori_adj, ori_attr = standardize(adj_matrix=graph['adj_matrix'],
                                 attr_matrix=graph['attr_matrix'])
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
    else:
        # Remove-Add
        num = nodepair_num

    ori_ppr_changing = pagerank_adj_changing(
        ori_adj, alpha, fragile, cur_controlled, local_budget, logits, global_budget, upper_bounds)
    np.save(output_file + '_ori_ppr_changing.npy', ori_ppr_changing)

    con_ratio = 0.05
    cur_con_num = int(round(edge_num * (con_ratio-0.05)))
    con_budget = int(round(edge_num * con_ratio))
    cur_con_num = int(round(edge_num * (con_ratio-0.05)))
    # initilization
    cur_controlled = torch.ones((n,n))
    adj_controlled = grad_adv_immune(adj, ori_ppr_changing, cur_controlled, exceed_local, cur_con_num, con_budget, logits, labels, alpha, con_budget_local)

    for con_ratio in np.arange(0.1, 0.15, 0.05):
        cur_con_num = int(round(edge_num * (con_ratio-0.05)))
        con_budget = int(round(edge_num * con_ratio))

        print('After control {:d}:'.format(con_budget))
        cur_control = np.aload('citeseer/add_rem/global_local_6/adj_controlled_%.1f%%.npy'%(con_ratio-0.05)*100)
        cur_controlled = torch.Tensor(cur_control)

        exceed_local = []
        for exceed_i in range(n):
            cur_localcon = np.where(cur_control[exceed_i]==0)[0].shape[0]
            if cur_localcon == con_budget_local[exceed_i]:
                exceed_local.append(exceed_i)

        adj_controlled = continue_grad_adv_immune(adj, ori_ppr_changing, cur_controlled, exceed_local, cur_con_num, con_budget, logits, labels, alpha, con_budget_local)

    
    for idx in range(n):
        tmp = np.where(adj_controlled[idx] == 0)[0].shape[0]
        if tmp > con_budget_local[idx]:
            print('Node %d: actual %d, budget: %d' % (idx,tmp,con_budget_local[idx]))
            
            
    con_ppr_changing = pagerank_adj_changing(ori_adj, alpha, fragile, adj_controlled, local_budget, logits)
    con_worst_margins = worst_margins_given_k_squared(con_ppr_changing, labels, logits)
    print('After control {:d}: Ratio of certified all nodes: {:.6f}'.format(con_budget, (con_worst_margins>0).mean()))
    print('num of robust node:', (con_worst_margins>0).sum())
    print(con_worst_margins.mean())

    adj_controlled = adj_controlled.detach().numpy()
    np.save('my_citeseer/add_rem/global_local_6/adj_controlled_%.1f%%.npy'%(con_ratio*100), adj_controlled)






if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AdvImmune')

    # configure
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--scenario', choices=['rem','add_rem'], default='rem', help='Scenarios of surrogate attack model')

    # dataset
    parser.add_argument('--dataset', choices=['citeseer','cora','reddit'], default='citeseer',
                        help='dataset to use')
    
    
    parser.add_argument('--suffix', type=str, default='_',
                        help='suffix of the checkpoint')
    
    # budgets
    parser.add_argument('--alpha', default=0.85,help='alpha in personalized pagerank')
    parser.add_argument('--attackLocal', type=int, default=0, help='the local budget of surrogate model and robustness certification')
    parser.add_argument('--immuneLocal', type=int, default=0, help='the local budget of immunization')
#     parser.add_argument('--immuneGlobal', default=0.05, help='the global budget of immunization')
    
    args = parser.parse_args()
    opts = args.__dict__.copy()
    att_sucess = main(opts) 

        

