import torch
import numpy as np
import pickle
import matplotlib.pylab as plt
import sys
import os
import time
import random_initialize as ri

with open('80000_Ca_signal_500ms.pickle','rb+') as f:
    Y=pickle.load(f)
with open('80000_Ca_signal_500ms2.pickle','rb+') as f:
    Ct=pickle.load(f)
with open('80000_Ca_signal_property.pickle','rb+') as f:
    Ca_property=pickle.load(f)
with open('80000Vi.pickle','rb+') as f:
    real_V_i,real_active_sum,real_gu_1,real_gu_3=pickle.load(f)
torch.set_default_tensor_type(torch.FloatTensor)
if torch.cuda.is_available():
    # device = torch.device("cuda")
    device = torch.device("cpu")

#导入节点信息与连接信息
path = './'
property, w_uij = ri.connect_for_block(path)
property=property.to(device)
w_uij=w_uij.to(device)
N, K=1,1000
property = property.reshape([N * K, -1])
real_gu_i = property[:,10:14].mean(axis=0)#获取正确的gu_i

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
        #if isinstance(H, torch.cuda.sparse.FloatTensor):#原来为torch.sparse.FloatTensor或torch.sparse.Tensor，这里修改了
        return torch.stack([torch.sparse.mm(H[i], b.unsqueeze(1)).squeeze(1) for i in range(4)])
        #else:
            #return torch.matmul(H, b.unsqueeze(0).unsqueeze(2)).squeeze(2)

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
    return (a*torch.exp(-x)+b)/(1+torch.exp(-x))
#@jit(nopython=True)
def inv_sigmoid_func(x,a,b):
    assert len(x[x<=a])==0 and len(x[x>=b])==0
    #d = (b-a)/1000
    #x[x<=a] = a+d #截断
    #x[x>=b] = b-d #截断
    return torch.log((x-a)/(b-x))

def update(sample_index):
    Block = B_list[sample_index]
    Block.g_ui[0, :] = sigmoid_func(all_cell_states[sample_index, :, 0], gu_1_low_bound, gu_1_upper_bound)
    Block.g_ui[2, :] = sigmoid_func(all_cell_states[sample_index, :, 1], gu_3_low_bound, gu_3_upper_bound)
    # 更新系统状态
    sum_activate, mean_Vi = Block.run(noise_rate=0.007)
    # 收集统计信息
    all_sum_activate[t, sample_index] = sum_activate
    #all_mean_Vi[t, sample_index] = mean_Vi
    all_Vi[t, sample_index, :] = Block.V_i
    if t % obs_interval == 0:  # 观测节点
        a = sigmoid_func(all_cell_states[sample_index, 1, 2], a_low_bound, a_upper_bound)  ###所有cell共享一个a,b,yj
        b = sigmoid_func(all_cell_states[sample_index, 1, 3], b_low_bound, b_upper_bound)
        yj = sigmoid_func(all_cell_states[sample_index, 1, 4], yj_low_bound, yj_upper_bound)
        Ca_property_list[t, sample_index, 0] = a
        Ca_property_list[t, sample_index, 1] = b
        Ca_property_list[t, sample_index, 2] = yj
        V_i_j_ = all_Vi[t - obs_interval:t, sample_index, :]
        all_Ct[t, sample_index, :] = yj * all_Ct[t - 1, sample_index, :] + (V_i_j_ >= V_th).sum(axis=0)
        all_Yt[t, sample_index, :] = a * (all_Ct[t, sample_index, :] + b)  # +epslo[i,:]
        all_cell_states[sample_index, :, state_size - 1] = all_Yt[t, sample_index, :]

#initialization state: gu_1, gu_3, a,b, yj, Y
delta_t =1
T_size = 80000#用于调节时间长度
obs_interval=200
state_size = 6
samples_size = 10
V_th = -50;V_reset = -65
mu0=0#mu0=-3
gu_i_C0 = 1**2
C0 = 1**2
sample_sigma = 0.01
ratio = 3;ratio2 = 4
bounds=np.array([[real_gu_i[0]/ratio,real_gu_i[0]*ratio],[real_gu_i[2]/ratio,real_gu_i[2]*ratio],\
                 [Ca_property[0]/ratio2,Ca_property[0]*ratio2],[Ca_property[1]/ratio2,Ca_property[1]*ratio2],\
                 [Ca_property[2]/ratio2,Ca_property[2]*ratio2]])

#bounds =bounds.to(device)
gu_1_low_bound,gu_1_upper_bound = bounds[0,:]
gu_3_low_bound,gu_3_upper_bound = bounds[1,:]
a_low_bound,a_upper_bound = bounds[2,:]
b_low_bound,b_upper_bound = bounds[3,:]
yj_low_bound,yj_upper_bound = bounds[4,:]

all_cell_states = torch.zeros([samples_size,K,state_size],device=device) #所有sample的cell的参数全体
all_cell_states[:,:,0] = torch.tensor(np.random.normal(0,gu_i_C0,[samples_size,K])) #gu_1
all_cell_states[:,:,1] = torch.tensor(np.random.normal(0,gu_i_C0,[samples_size,K])) #gu_3
all_cell_states[:,:,2:5] = torch.tensor(np.random.normal(0,C0,[samples_size,K,3]))
all_cell_states[:,:,5] = 0
Ca_property_list = torch.zeros([T_size,samples_size,3],device=device)

B_list = []
for sample_index in range(samples_size):
    #初始化各block的神经元状态
    property_=property[:]
    property_[:,10] = sigmoid_func(all_cell_states[sample_index,:,0],gu_1_low_bound,gu_1_upper_bound)
    property_[:,12] = sigmoid_func(all_cell_states[sample_index,:,1],gu_3_low_bound,gu_3_upper_bound)
    B = block(
    node_property=property_,
    w_uij=w_uij,
    delta_t=delta_t,
)
    B_list.append(B)

#undate
cells_state_size = state_size*K
diag_list = np.ones(cells_state_size)
Sigma_z =np.diag(diag_list**2)
gama_num = 1**2
gama = torch.tensor(gama_num,device=device)

gain_weight_ = 0.2
other_gain_weight = (1-gain_weight_)/(K-1)
gain_weight = torch.ones([samples_size,K,state_size],device=device)*other_gain_weight

all_sum_activate = torch.zeros([T_size,samples_size],device=device)
all_mean_Vi = torch.zeros([T_size,samples_size],device=device)
all_mean_Vi[0,:] = (V_th+V_reset)/2
all_Vi = torch.zeros([T_size,samples_size,K],device=device)
all_Vi[0,:,:] = (V_th+V_reset)/2
all_Ct = torch.zeros([T_size,samples_size,K],device=device)
all_Yt = torch.zeros([T_size,samples_size,K],device=device)
H = torch.zeros([1,cells_state_size],device=device);H[0,cells_state_size-1]=1

gain_weight_list = torch.zeros([K,samples_size,K,state_size])
for cell_index in range(K):
    gain_weight_copy = gain_weight[:]
    gain_weight_copy[:, cell_index, :] = gain_weight_
    gain_weight_list[cell_index,:,:,:]=gain_weight_copy
gain_weight_list = gain_weight_list.to(device)
samples_obs_all=Y[:T_size,:].unsqueeze(2)+torch.normal(0,gama_num,[T_size,K,samples_size])#观测量的分析
samples_obs_all = samples_obs_all.permute([0,2,1])
samples_obs_all=samples_obs_all.to(device)

if __name__ == '__main__':
    start = time.time()
    for t in range(T_size):
        if t%10 == 0:
            print("t:{}".format(t))
        for sample_index in range(samples_size):
            update(sample_index)
        if t%obs_interval==0:#观测节点,Kalman update
            error = samples_obs_all[t, :, :] - all_cell_states[:, :, -1]
            mu_cell = all_cell_states.mean(dim=0).reshape(-1, 1)# 注意是否存在乱序
            cells = all_cell_states.reshape(samples_size, -1).T  # 注意是否存在乱序
            cells += torch.tensor(np.random.normal(0, sample_sigma, [cells_state_size, samples_size]),
                                  device=device)  # draw #是否可以只draw一次
            deviation = cells - mu_cell
            half_cov = torch.matmul(deviation[-1, :].reshape(-1, samples_size), deviation.T) / (samples_size - 1)
            K_kf = half_cov.T / (torch.matmul(half_cov, H.T) + gama)
            K_kf_ = torch.stack([K_kf.T] * K, dim=0)
            error_ = error.T.unsqueeze(2)
            K_kf_error = torch.bmm(error_, K_kf_)
            K_kf_error_shaped = K_kf_error.reshape([K, samples_size, K, state_size])
            gain_list = K_kf_error_shaped * gain_weight_list
            all_gain = gain_list.sum(dim=0)

            all_gain[all_gain > 1] = 1;all_gain[all_gain < -1] = -1
            all_cell_states += all_gain
    end = time.time()

    print("cost time: {}".format(end-start))

    Ca_property_list_obs = np.zeros([T_size//obs_interval,4])
    Y_obs = np.zeros(T_size//obs_interval)
    for i in range(T_size):
        if i%obs_interval==0:
            Ca_property_list_obs[i//obs_interval, 0] = Ca_property_list[i, : ,0].mean().to("cpu").numpy()
            Ca_property_list_obs[i//obs_interval, 1] = Ca_property_list[i, :, 1].mean().to("cpu").numpy()
            Ca_property_list_obs[i//obs_interval, 2] = Ca_property_list[i, :, 2].mean().to("cpu").numpy()
            Ca_property_list_obs[i // obs_interval, 3] = all_Yt[i,:,:].mean(dim=0).sum().to("cpu").numpy()
            Y_obs[i // obs_interval] = Y[i+1, :].sum()


    plt.figure(1)
    plt.subplot(221)
    plt.scatter(range(K),sigmoid_func(all_cell_states[:,:,0].mean(dim=0).to("cpu"),gu_1_low_bound,gu_1_upper_bound))
    plt.scatter(range(K),real_gu_1)
    plt.legend(["gu_1","real_gu_1"])
    plt.title("gu_1")

    plt.subplot(222)
    plt.hist(sigmoid_func(all_cell_states[:,:,0].mean(dim=0).to("cpu"),gu_1_low_bound,gu_1_upper_bound)-real_gu_1, bins=100)
    plt.title("gu_1 error")

    plt.subplot(223)
    plt.scatter(range(K),sigmoid_func(all_cell_states[:,:,1].mean(dim=0).to("cpu"),gu_3_low_bound,gu_3_upper_bound))
    plt.scatter(range(K),real_gu_3)
    plt.legend(["gu_3","real_gu_3"])
    plt.title("gu_3")

    plt.subplot(224)
    plt.hist(sigmoid_func(all_cell_states[:,:,1].mean(dim=0).to("cpu"),gu_3_low_bound,gu_3_upper_bound)-real_gu_3, bins=100)
    plt.title("gu_3 error")

    plt.figure(2)
    plt.subplot(311)
    plt.plot(all_sum_activate.mean(dim=1).to("cpu"))
    plt.plot(real_active_sum[:T_size])
    plt.legend(["assimilation","real_active_sum"])
    plt.title("active_sum")
    plt.xlabel("t ms")
    plt.ylabel("active_sum")

    plt.subplot(312)
    plt.plot(all_Vi[:T_size,:,:].mean(dim=[1,2]).to("cpu"))
    plt.plot(real_V_i[:T_size,:].mean(axis=1))
    plt.legend(["assimilation","real_V_mean"])
    plt.title("V")
    plt.xlabel("t ms")
    plt.ylabel("V")

    plt.subplot(313)
    plt.scatter(list(range(T_size//obs_interval)),Ca_property_list_obs[:,3])
    plt.scatter(list(range(T_size//obs_interval)),Y_obs)
    plt.legend(["assimilation", "real_Y"])
    plt.title("Y")

    plt.figure(3)
    plt.subplot(311)
    plt.scatter(list(range(T_size//obs_interval)),Ca_property_list_obs[:,0])
    plt.axhline(y=Ca_property[0],color = "red")
    plt.xlabel("times of observation")
    plt.legend(["assimilation", "real_a"])
    plt.title("a")

    plt.subplot(312)
    plt.scatter(list(range(T_size//obs_interval)),Ca_property_list_obs[:,1])
    plt.axhline(y=Ca_property[1],color = "red")
    plt.xlabel("times of observation")
    plt.legend(["assimilation", "real_b"])
    plt.title("b")

    plt.subplot(313)
    plt.scatter(list(range(T_size//obs_interval)),Ca_property_list_obs[:,2])
    plt.axhline(y=Ca_property[2],color = "red")
    plt.xlabel("times of observation")
    plt.legend(["assimilation", "real_yj"])
    plt.title("yj")
    plt.show()
