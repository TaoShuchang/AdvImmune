import torch
from torch import nn
from utils import *
from attack import *
import random
import threading

import numpy as np
from tqdm import tqdm
import math
import scipy.sparse as sp

import time  

class MyThread(threading.Thread):
    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

# 只用一个图，所以方法中保存元梯度信息
def grad_adv_immune(adj, ppr_adj_changing, exceed_local, con_budget, logits, labels, alpha, con_budget_local=None, grad_flag=False, rand=None):
    """
    ori_adj: sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    
    Return:
    """
    begin = time.time()
    n,nc = logits.shape
    control_num = 0   

    adj_controlled = torch.ones([n, n])
    adj_controlled.requires_grad_()
    control_10 = []
    while control_num < con_budget:
        # 需不需要考虑局部控制
        local_flag = True
        if control_num % 50 == 0:
            print('control edges num:', control_num)
        # 控制边后，扰动的新矩阵  
        # A^ = A + A' * Ec
        ppr_flipped = {}
        loss = {}
        for c1 in range(nc):
            for c2 in range(nc):
                if c1 != c2:
                    modified_adj = adj + torch.mul(ppr_adj_changing[(c1,c2)]['changing'], adj_controlled)
                    ppr_flipped[(c1,c2)] = propagation_matrix(adj=modified_adj, alpha=alpha)
                    tmp_re = torch.from_numpy(logits[:,c1] - logits[:,c2]).float()
                    loss[c1,c2] = ppr_flipped[(c1,c2)] @ tmp_re

        worst_class = worstcase_class(ppr_flipped, labels, logits)
        final_loss = compute_final_loss(loss, labels, worst_class)

        # 判断是梯度全1还是有0有1
        # 计算 pi*r 对于 adj_controlled 元梯度
        if grad_flag is not False:
            robust = (final_loss<0).float()
            b = torch.zeros_like(final_loss)
            final_loss = torch.where(final_loss < 0, final_loss, b)
        final_loss.backward(torch.ones_like(final_loss))
        adj_grad = -adj_controlled.grad

        # Get argmax of the meta gradients.
        # 选一个最大的元梯度，小心梯度最大的永远都是同一个！！  
        if control_num != 0:
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
        
        # local_flag = False
        # 给控制的边集增加一条边
        adj_controlled[new_row_idx.item(), new_col_idx.item()] = 0
        if rand is not None:
            control_10.append([new_row_idx.item(), new_col_idx.item()])
            if len(control_10) % 10 == 0:
                for con in control_10:
                    ran = random.random()
                    if ran < rand:
                        adj_controlled[con[0], con[1]] = 1
                    print([con[0], con[1]], ran, adj_controlled[con[0], con[1]].item())
                control_10 = []
        # print(new_row_idx, new_col_idx)
        # 重新变成叶子节点，有梯度
        adj_controlled = adj_controlled.detach()
        adj_controlled.requires_grad_()
        
        row_idx = np.where(adj_controlled == 0)[0]
        col_idx = np.where(adj_controlled == 0)[1]
        control_num = row_idx.shape[0]
        
        del modified_adj, ppr_flipped
        
    del adj_grad, adj_grad_argmax, row_idx, col_idx, exceed_local
    end = time.time()
    print('time: ', end - begin)
    return adj_controlled


def compute_loss(adj, ppr_adj_changing, adj_controlled, logits, alpha, c1, c2):
    modified_adj = adj + torch.mul(ppr_adj_changing, adj_controlled)
    ppr_flipped = propagation_matrix(adj=modified_adj, alpha=alpha)
    tmp_re = torch.from_numpy(logits[:,c1] - logits[:,c2]).float()
    loss_each_class = ppr_flipped @ tmp_re
#     loss_each_class.backward(torch.ones_like(loss_each_class))
#     grad_each_class = adj_controlled.grad
    
    return c1, c2, ppr_flipped, loss_each_class


def continue_grad_adv_immune(adj, ppr_adj_changing, cur_adj_controlled, exceed_local, cur_control_num, con_budget, logits, labels, alpha, 
                        con_budget_local=None, grad_flag=False, rand=None):
    """
    adj: float tensor, shape [n, n]
        Sparse adjacency matrix.
    
    Return:
    """
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
        # A^ = A + A' * Ec
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

#         every_cnt = min(10, con_budget-control_num)
#         while cnt < every_cnt:
#             adj_grad_argmax = torch.argmax(adj_grad)
#             new_row_idx, new_col_idx = unravel_index(adj_grad_argmax, adj.shape)
#             adj_grad[new_row_idx, new_col_idx] = 0
#             adj_controlled[new_row_idx.item(), new_col_idx.item()] = 0
#             cnt += 1
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
#         if (control_num-cur_control_num)>200 and (control_num-cur_control_num) %100 ==0:
#             np.save('my_reddit/add_rem/one_local_%d/adj_controlled_%d.npy'%(6,int(control_num-cur_control_num)), adj_controlled.detach().numpy())
        del ppr_flipped

    del adj_grad, adj_grad_argmax, row_idx, col_idx
    end = time.time()
    print('time: ', end - begin)
    
    return adj_controlled


def compute_grad(adj, ppr_adj_changing, cur_adj_controlled, logits, labels, alpha, 
                        con_budget_local=None, grad_flag=False, rand=None):
    """
    adj: float tensor, shape [n, n]
        Sparse adjacency matrix.
    
    Return:
    """
    begin = time.time()
    n,nc = logits.shape

    adj_controlled = cur_adj_controlled.clone()
    adj_controlled.requires_grad_()
    # 需不需要考虑局部控制
    local_flag = True
        # 控制边后，扰动的新矩阵  
        # A^ = A + A' * Ec
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
    print(torch.nonzero(adj_grad).shape)
    return adj_grad


def k_parallel_continue_immune(adj, alpha, ppr_adj_changing, cur_adj_chontrolled, cur_control_num, con_budget, logits):
    parallel = Parallel(10)

    n, nc = logits.shape
    
    results = parallel(delayed(continue_adv_immune)(
        adj, ppr_adj_changing, cur_adj_chontrolled, cur_control_num, con_budget, logits, c1, c2, alpha)
                       for c1 in range(nc)
                       for c2 in range(nc)
                       if c1 != c2)

    k_squared_controlled = {}
    for c1, c2, adj_controlled in results:
        k_squared_controlled[(c1, c2)] = {'control': adj_controlled}
    del results
    return k_squared_controlled

