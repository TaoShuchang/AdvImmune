import torch
import random
import threading
import time
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
from joblib import Parallel, delayed
from collections import Counter
from utils import propagation_matrix, flip_edges, unravel_index
from SurrogateAttack import policy_iteration

class MyThread(threading.Thread):
    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  
        except Exception:
            return None


def grad_adv_immune(adj, ppr_adj_changing, cur_adj_controlled, exceed_local, cur_control_num, con_budget, logits, labels, alpha, con_budget_local=None, grad_flag=False, rand=None):
    
    begin = time.time()
    n,nc = logits.shape

    adj_controlled = cur_adj_controlled.clone()
    adj_controlled.requires_grad_()
    control_num = cur_control_num
    while control_num < con_budget:
        # 需不需要考虑局部控制
        local_flag = True
        if (control_num-cur_control_num) % 20 == 0:
            print('control edges num:', control_num)
        # 控制边后，扰动的新矩阵  
        # A^ = A + A' * Ac
        cnt = 0
        ppr_flipped = {}
        loss = {}
        threads = []
        for c1 in range(nc):
            a = np.arange(nc)
            new_a = np.delete(a, c1)
            for c2 in new_a:
                p = MyThread(compute_loss, args=(adj, ppr_adj_changing[(c1,c2)]['changing'], 
                                                     adj_controlled, logits, alpha, c1, c2))
                p.start()
                threads.append(p)

        for p in threads:
            p.join()
            c1, c2, ppr_flipped_class, loss_each_class = p.get_result()
            ppr_flipped[(c1, c2)] = ppr_flipped_class
            loss[c1,c2] = loss_each_class
    
        worst_class = worstcase_class(ppr_flipped, labels, logits)
        final_loss = compute_final_loss(loss, labels, worst_class)
        # 计算 pi*r 对于 adj_controlled 元梯度
        final_loss.backward(torch.ones_like(final_loss))
        adj_grad = -adj_controlled.grad
        # Get argmax of the meta gradients.
        # 选一个最大的元梯度，小心梯度最大的永远都是同一个！！
        if control_num != 0:
            row_idx = np.where(adj_controlled == 0)[0]
            col_idx = np.where(adj_controlled == 0)[1]
            adj_grad[row_idx, col_idx] = 0
            for idx in exceed_local:
                adj_grad[idx, :] = 0

        if con_budget_local is not None:
            while local_flag == True:
            # local_flag = True
                adj_grad_argmax = torch.argmax(adj_grad)
                new_row_idx, new_col_idx = unravel_index(adj_grad_argmax, adj.shape)
                if np.where(adj_controlled[new_row_idx.data] < 1)[0].shape[0] >= con_budget_local[new_row_idx.data]:
                    adj_grad[new_row_idx.data,:] = 0
                    local_flag = True
                    exceed_local.append(new_row_idx.data)
                else:
                    local_flag = False
        else:
            adj_grad_argmax = torch.argmax(adj_grad)
            new_row_idx, new_col_idx = unravel_index(adj_grad_argmax, adj.shape)
        
        adj_controlled[new_row_idx.item(), new_col_idx.item()] = 0
        
        # 重新变成叶子节点，有梯度
        adj_controlled = adj_controlled.detach()
        adj_controlled.requires_grad_()
        
        row_idx = np.where(adj_controlled == 0)[0]
        col_idx = np.where(adj_controlled == 0)[1]

        control_num = row_idx.shape[0]
        del ppr_flipped

    del adj_grad, adj_grad_argmax, row_idx, col_idx
    end = time.time()
    print('time: ', end - begin)
    
    return adj_controlled


def compute_loss(adj, ppr_adj_changing, adj_controlled, logits, alpha, c1, c2):
    modified_adj = adj + torch.mul(ppr_adj_changing, adj_controlled)
    ppr_flipped = propagation_matrix(adj=modified_adj, alpha=alpha)
    tmp_re = torch.from_numpy(logits[:,c1] - logits[:,c2]).float()
    loss_each_class = ppr_flipped @ tmp_re
    
    return c1, c2, ppr_flipped, loss_each_class


def compute_final_loss(loss, labels, worst_class):
    n = labels.shape[0]
    final_loss = torch.unsqueeze(loss[labels[0], worst_class[0]][0], 0)
    for i in range(1,n):
        tmp = torch.unsqueeze(loss[labels[i], worst_class[i]][i], 0)
        final_loss = torch.cat((final_loss, tmp), 0)

    return final_loss

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


def worst_margin_local(ori_adj, alpha, fragile, adj_controlled, local_budget, logits, true_class, other_class):
    # min(true - other_class) = -max(other_class-true_class)
    
    reward = logits[:, other_class] - logits[:, true_class]

    # pick any teleport vector since the optimal fragile edges do not depend on it
    teleport = np.zeros_like(reward)
    teleport[0] = 1
    # controlled_fragile = np.array([edge.data for edge in fragile if adj_controlled[edge[0]][edge[1]] == 1])

    control = torch.nonzero(adj_controlled==0).numpy()
    con_tuple = [(edge[0],edge[1]) for edge in control]
    fra_tuple = [(edge[0],edge[1]) for edge in fragile]

    diff_cnt =  Counter(fra_tuple) - Counter(con_tuple)
    new_fragile = np.array([list(k) for k,v in diff_cnt.items() if v == 1])
    opt_fragile, obj_value = policy_iteration(
        adj=ori_adj, alpha=alpha, fragile=new_fragile, local_budget=local_budget, reward=reward, teleport=teleport)
    
    # 改动的地方
    adj = torch.from_numpy(ori_adj.todense()).float()
    adj_changing = compute_adj_changing(adj, opt_fragile)
    # modified_adj = adj + torch.mul(adj_changing, adj_controlled)
    adj_flipped = flip_edges(ori_adj, opt_fragile)
    adj_flipped = torch.from_numpy(adj_flipped.todense()).float()

    ppr_flipped = propagation_matrix(adj=adj_flipped, alpha=alpha)

    return true_class, other_class, ppr_flipped, adj_changing

def compute_adj_changing(adj, opt_fragile):
    n = adj.shape[0]
    adj_changing = torch.zeros([n, n])

    for edge in opt_fragile:
        adj_changing[edge[0],edge[1]] = 1 - 2*adj[edge[0],edge[1]]
    del adj, opt_fragile
    return adj_changing


def pagerank_adj_changing(ori_adj, alpha, fragile, adj_controlled, local_budget, logits, global_budget=None, upper_bounds=None):
    parallel = Parallel(10)

    n, nc = logits.shape
    results = parallel(delayed(worst_margin_local)(
            ori_adj, alpha, fragile, adj_controlled, local_budget, logits, c1, c2)
                        for c1 in range(nc)
                        for c2 in range(nc)
                        if c1 != c2)
    ppr_adj_changing= {}
    for c1, c2, ppr_flipped, adj_changing in results:
        ppr_adj_changing[(c1, c2)] = {'ppr':ppr_flipped, 'changing': adj_changing}
    
    return ppr_adj_changing


def worst_margins_given_k_squared(ppr_adj_changing, labels, logits):
    """
    Computes the exact worst-case margins for all node via the PageRank matrix of the perturbed graphs.
    Parameters
    """
    n, nc = logits.shape
    worst_margins_all = np.ones((nc, nc, n)) * np.inf

    for c1 in range(nc):
        for c2 in range(nc):
            if c1 != c2:
                worst_margins_all[c1, c2] = (ppr_adj_changing[c1, c2]['ppr'].detach().numpy() @ (logits[:, c1] - logits[:, c2]))

    # selected the reference label according to the labels vector and find the minimum among all other classes
    worst_margins = np.nanmin(worst_margins_all[labels, :, np.arange(n)], 1)

    return worst_margins
