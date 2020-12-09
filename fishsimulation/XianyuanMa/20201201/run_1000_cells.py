import matplotlib.pylab as plt
import sys
import numpy as np
import os
import time
import random_initialize as ri
import torch
import pickle

class block:
    def __init__(self, node_property, w_uij, delta_t=1):
        # A block is a set of spliking neurals with inner full connections, we consider 4 connections:
        # AMPA, NMDA, GABAa and GABAb
        # shape note:
        #
        # N: numbers of neural cells
        # K: connections kind, = 4 (AMPA, NMDA, GABAa and GABAb)
        assert len(w_uij.shape) == 3
        N = w_uij.shape[1]#神经元的数量
        K = w_uij.shape[0]#K=4，4种神经递质

        self.w_uij = w_uij  # shape [K, N, N] #w_uij[a,b,c]->c号神经元对b号神经元的a种神经递质的连接权值
        self.delta_t = delta_t

        self.update_property(node_property)

        self.t_ik_last = torch.zeros([N], device=self.w_uij.device) # shape [N]
        self.V_i = torch.ones([N], device=self.w_uij.device) * (self.V_th + self.V_reset)/2  # membrane potential, shape: [N]
        self.J_ui = torch.zeros([K, N], device=self.w_uij.device)  # shape [K, N]
        self.t = torch.tensor(0., device=self.w_uij.device)  # scalar

        self.update_I_syn()

    @staticmethod
    def expand(t, size):
        t = torch.tensor(t)
        shape = list(t.shape) + [1] * (len(size) - len(t.shape))
        return t.reshape(shape).expand(size)

    def update_J_ui(self, delta_t, active):
        # active shape: [N], dtype bool
        # t is a scalar
        self.J_ui = self.J_ui * torch.exp(-delta_t / self.tau_ui)
        J_ui_activate_part = self.bmm(self.w_uij, active.float()) # !!! this part can be sparse.
        self.J_ui += J_ui_activate_part
        pass

    @staticmethod
    def bmm(H, b):
        if isinstance(H, torch.sparse.FloatTensor):#torch.sparse.Tensor
            return torch.stack([torch.sparse.mm(H[i], b.unsqueeze(1)).squeeze(1) for i in range(4)])
        else:
            return torch.matmul(H, b.unsqueeze(0).unsqueeze(2)).squeeze(2)

    def update_I_syn(self):
        self.I_ui = self.g_ui * (self.V_ui - self.V_i) * self.J_ui
        # [K, N]            [K, N] - [K, 1]
        self.I_syn = self.I_ui.sum(dim=0)
        pass

    def update_Vi(self, delta_t):
        main_part = -self.g_Li * (self.V_i - self.V_L)
        C_diff_Vi = main_part + self.I_syn + self.I_extern_Input
        delta_Vi = delta_t / self.C * C_diff_Vi

        Vi_normal = self.V_i + delta_Vi

        # if t < self.t_ik_last + self.T_ref:
        #   V_i = V_reset
        # else:
        #   V_i = Vi_normal
        is_not_saturated = (self.t >= self.t_ik_last + self.T_ref)
        V_i = torch.where(is_not_saturated, Vi_normal, self.V_reset)
        #print(is_not_saturated.sum())
        active = (V_i >= self.V_th)#判断是否激发
        self.V_i = torch.min(V_i, self.V_th)
        return active

    def update_t_ik_last(self, active):
        self.t_ik_last = torch.where(active, self.t, self.t_ik_last)

    def run(self, noise_rate=0.01, isolated=False):
        self.t += self.delta_t#时间步进
        self.active = self.update_Vi(self.delta_t)#状态是否激发
        if not isolated:#孤立设置
            new_active = (torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < noise_rate) | self.active
        else:#非孤立设置
            new_active = (torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < noise_rate)
        self.update_J_ui(self.delta_t, new_active)
        self.update_I_syn()
        self.update_t_ik_last(self.active)

        mean_Vi = []
        sum_activate = []
        for i in range(self.sub_idx.max().int() + 1):
            mean_Vi.append(self.V_i[self.sub_idx == i].mean())
            sum_activate.append(self.active[self.sub_idx == i].float().sum())

        return torch.stack(sum_activate), torch.stack(mean_Vi)

    def update_property(self, node_property):
        # update property
        # column of node_property is
        # E/I, blocked_in_stat, has_extern_Input, no_input, C, g_Li, V_L, V_th, V_reset, g_ui, V_ui, tau_ui
        E_I, blocked_in_stat, I_extern_Input, sub_idx, C, T_ref, g_Li, V_L, V_th, V_reset, g_ui, V_ui, tau_ui = \
            node_property.transpose(0, 1).split([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4])

        self.I_extern_Input = I_extern_Input.squeeze(0) # extern_input index , shape[K]
        self.V_ui = V_ui  # AMPA, NMDA, GABAa and GABAb potential, shape [K, N]
        self.tau_ui = tau_ui  # shape [K, N]
        self.g_ui = g_ui  # shape [K, N]
        self.g_Li = g_Li.squeeze(0)  # shape [N]
        self.V_L = V_L.squeeze(0)  # shape [N]
        self.C = C.squeeze(0)   # shape [N]
        self.sub_idx = sub_idx.squeeze(0) # shape [N]
        self.V_th = V_th.squeeze(0)   # shape [N]
        self.V_reset = V_reset.squeeze(0)  # shape [N]
        self.T_ref = T_ref.squeeze(0) # shape [N]
        return True

    def update_conn_weight(self, conn_idx, conn_weight):
        # update part of conn_weight
        # conn_idx shape is [4, X']
        # conn_weight shape is [X']
        self.w_uij[conn_idx] = conn_weight
        return True

#@jit(nopython=True)
def sigmoid_func(x,a,b):
    assert a<b
    return (a+b*torch.exp(x))/(1+torch.exp(x))
#@jit(nopython=True)
def inv_sigmoid_func(x,a,b):
    #assert len(x[x<=a])==0 and len(x[x>=b])==0
    d = (b-a)/1000
    x[x<=a] = a+d #截断
    x[x>=b] = b-d #截断
    return torch.log((x-a)/(b-x))

path = './'
property, w_uij = ri.connect_for_block(path)
N, K=1,1000
property = property.reshape([N * K, -1])

#对gu_i进行扰动
property[:,10]= sigmoid_func(torch.tensor(np.random.normal(inv_sigmoid_func(property[1,10],property[1,10]/3,property[1,10]*3),1**2,[1000])),property[1,10]/3,property[1,10]*3)
property[:,12]= sigmoid_func(torch.tensor(np.random.normal(inv_sigmoid_func(property[1,12],property[1,12]/3,property[1,12]*3),1**2,[1000])),property[1,12]/3,property[1,12]*3)

#设置block，输入节点信息和连接信息
B = block(
    node_property=property,
    w_uij=w_uij,
    delta_t=1,
)

T_size = 80000
all_sum_activate = torch.zeros([T_size])
V_i = []
for k in range(T_size):
    sum_activate, mean_Vi = B.run(noise_rate=0.007)
    all_sum_activate[k] = sum_activate
    # if k%100 ==0:
    #     print(k, int(active.sum()))#打印时刻以及该时刻的神经元激发数量
    V_i.append(B.V_i.numpy())

V_i = np.array(V_i)

with open('80000Vi.pickle','wb') as f:
    pickle.dump([V_i,all_sum_activate,B.g_ui[0,:],B.g_ui[2,:]],f)