import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

def generate_property(C=1, T_ref=5, g_Li=0.001, V_Ls=-75, V_th=-50, V_reset=-65,
                      g_ui=np.array([10/500, 5/6000, 3/60, 1/1000]), V_ui=np.array([0, 0, -70, -100]),
                      tao_ui =np.array ([2, 40, 10, 50])):
    """
    Generate property

    :param C:
    :param T_ref:
    :param g_Li:
    :param V_Ls:
    :param V_th:
    :param V_reset:
    :param g_ui:
    :param V_ui:
    :param tao_ui:
    :return:
    """
    import numpy as np
    property = np.zeros([1, 22], dtype=np.float32)

    property[:, 0] = 1  # E/I
    property[:, 1] = 0  # blocked_in_stat
    property[:, 2] = 0  # I_external_Input
    property[:, 3] = 0  # sub_blocak_idx
    property[:, 4] = C  # 膜电容，一般1
    property[:, 5] = T_ref  # 不应期，1-10，原来为5
    property[:, 6] = g_Li  # 渗流突触导电率 一般为0.03
    property[:, 7] = V_Ls  # 渗流电位 一般为-75
    property[:, 8] = V_th  # 发放阈值电位 一般为-50
    property[:, 9] = V_reset  # 重置电位 一般为-65
    property[:, 10:14] = g_ui  # 神经递质电导
    property[:, 14:18] = V_ui  # 神经递质膜电流
    property[:, 18:22] = tao_ui  # 神经递质时间常数

    return property

def cov_calculator(samples, mu, state_size=9):
    sample_size = samples.shape[0]
    C = np.zeros([state_size,state_size])
    for i in range(sample_size):
        C+=np.multiply((samples[i].reshape(-1,1)-mu),(samples[i].reshape(-1,1)-mu).T)
    C = C/(sample_size-1)
    return C

def cov_calculator_tensor(samples, mu, state_size=9):
    sample_size = samples.shape[0]
    C = torch.zeros([state_size,state_size],device =device)
    for i in range(sample_size):
        C+=torch.matmul((samples[i].reshape(-1,1)-mu),(samples[i].reshape(-1,1)-mu).T)
    C = C/(sample_size-1)
    return C

def sigmoid_func(x, a, b):
    assert a<b
    return (a+b*np.exp(x))/(1+np.exp(x))

def inv_sigmoid_func(x,a,b):
    #  assert len(x[x<=a])==0 and len(x[x>=b])==0
    d = (b-a)/1000
    x[x<=a] = a+d #  截断
    x[x>=b] = b-d #  截断
    return np.log((x-a)/(b-x))

def inv_sigmoid_func_num(x,a,b):
    assert len(x[x<=a])==0 and len(x[x>=b])==0
    #d = (b-a)/1000
    #x = a+d if x<=a else x#截断
    #x = b-d if x>=b else x#截断
    return np.log((x-a)/(b-x))

def update_Jui_s(tau_ui,poisson,J_uiold,V_i,delta_t=1):
    #    tau_ui shape: 1*4
    J_ui =  J_uiold* np.exp(-delta_t / tau_ui.reshape(-1,4))+poisson.reshape(-1,4)
    return J_ui

def update_Vi_s(I_extern_Input,g_Li,g_ui,V_ui,V_L ,C,V_th,V_reset ,T_ref, J_ui,V_i,delta_t):
    I_ui = g_ui.reshape(-1,4)* (V_ui.reshape(-1,4) - V_i.reshape(-1,1)) * J_ui
    I_syn = I_ui.sum(axis=1)
    delta_Vi = delta_t / C * (-g_Li * (V_i - V_L) + I_syn + I_extern_Input)
    return V_i + delta_Vi

def initialize_state(delta_t,T_size,state_size,samples_size,bounds,mu0,C0):
    gu_1_low_bound,gu_1_upper_bound = bounds[0,:]
    gu_3_low_bound,gu_3_upper_bound = bounds[1,:]
    V_low_bound,V_upper_bound = bounds[2,:]
    a_low_bound,a_upper_bound = bounds[3,:]
    b_low_bound,b_upper_bound = bounds[4,:]
    yj_low_bound,yj_upper_bound = bounds[5,:]

    learn_states = np.random.multivariate_normal(mu0,C0,samples_size)#初始化
    #再映射回来
    learn_states[:,0] = sigmoid_func(learn_states[:,0],gu_1_low_bound,gu_1_upper_bound)
    learn_states[:,1] = sigmoid_func(learn_states[:,1],gu_3_low_bound,gu_3_upper_bound)
    learn_states[:,2] = sigmoid_func(learn_states[:,2],V_low_bound,V_upper_bound)
    learn_states[:,5] = sigmoid_func(learn_states[:,5],a_low_bound,a_upper_bound)
    learn_states[:,6] = sigmoid_func(learn_states[:,6],b_low_bound,b_upper_bound)
    learn_states[:,7] = sigmoid_func(learn_states[:,7],yj_low_bound,yj_upper_bound)

    auxilary_learn_states = np.zeros([samples_size,2])
    for i in range(samples_size):
        auxilary_learn_states[i,:] = np.array([-100,0])#[上一次的激发时间，是否激活]设成-100是为了防止一开始就静息
    Ju_i_states = np.zeros([samples_size,4])

    current_g_ui = np.zeros([samples_size,4])
    current_g_ui[:,0] = learn_states[:,0]
    current_g_ui[:,1] = real_g_ui[0,1]
    current_g_ui[:,2] = learn_states[:,1]
    current_g_ui[:,3] = real_g_ui[0,3]
    mu0 = learn_states.mean(axis=0)
    return learn_states,auxilary_learn_states,Ju_i_states,current_g_ui,mu0


def update_the_state(tau_ui,I_extern_Input,g_Li,g_ui,V_ui,V_L,C,V_th,V_reset,T_ref,real_g_ui,\
                     t,current_poisson,current_g_ui,last_states,last_auxilary_states,last_Ju_i_states):
    new_V_i = np.zeros(samples_size)
    auxilary_learn_states = last_auxilary_states
    learn_states = last_states
    V_i=learn_states[:,2]#V_i设为学习到的值
    t_reset = last_auxilary_states[:,0]+T_ref
    #接下来按照模型规则更新状态
    inside_mask = t_reset>t
    come_out_mask = ~inside_mask
    new_V_i[inside_mask] = V_reset
    auxilary_learn_states[inside_mask,1] = 0
    if come_out_mask.any():
        new_V_i[come_out_mask]=update_Vi_s(I_extern_Input,g_Li,current_g_ui,V_ui,V_L ,\
                                           C,V_th,V_reset ,T_ref, last_Ju_i_states,V_i,delta_t)[come_out_mask]
        over_th_mask = new_V_i>=V_th
        under_th_mask = ~over_th_mask
        new_V_i[over_th_mask]=V_th
        auxilary_learn_states[over_th_mask,1] = 1
        auxilary_learn_states[over_th_mask,0] = t
        auxilary_learn_states[under_th_mask,1] = 0
        learn_states[:,2] = new_V_i
    Ju_i_states=update_Jui_s(tau_ui,current_poisson,last_Ju_i_states,V_i,delta_t)
    return learn_states,auxilary_learn_states,Ju_i_states

def update_the_state_obs(learn_states,spike_sum):
    learn_states[:,3]=learn_states[:,7]*learn_states[:,3]+spike_sum
    learn_states[:,4]=learn_states[:,5]*(learn_states[:,3]+learn_states[:,6])#更新Yi
    return learn_states

def KF_update(learn_states,Sigma_z,bounds,state_size,gama,gama_num,H,i):
    learn_states[:,0] = inv_sigmoid_func(learn_states[:,0],bounds[0,0],bounds[0,1])
    learn_states[:,1] = inv_sigmoid_func(learn_states[:,1],bounds[1,0],bounds[1,1])
    learn_states[:,2] = inv_sigmoid_func(learn_states[:,2],bounds[2,0],bounds[2,1])
    learn_states[:,5] = inv_sigmoid_func(learn_states[:,5],bounds[3,0],bounds[3,1])
    learn_states[:,6] = inv_sigmoid_func(learn_states[:,6],bounds[4,0],bounds[4,1])
    learn_states[:,7] = inv_sigmoid_func(learn_states[:,7],bounds[5,0],bounds[5,1])
    learn_states[:,:]+=np.random.multivariate_normal(mu0,Sigma_z,samples_size)#draw
    #下面进行KF的矩阵更新
    learn_states_t = torch.tensor(learn_states[:,:]).to(device)
    mu = learn_states_t.mean(axis=0).reshape(-1,1)
    C_new = cov_calculator_tensor(learn_states_t,mu,state_size)
    K = torch.matmul(C_new,H.T)*(torch.matmul(torch.matmul(H,C_new),H.T)+gama)**(-1)
    samples_obs = torch.tensor(np.random.normal(Yt[i+1],gama_num,samples_size)).to(device)#观测量的分析
    learn_states+=torch.matmul(K,(samples_obs - learn_states_t[:,4]).reshape(1,-1)).T.to("cpu").numpy()#直接利用观察矩阵H
    #再映射回来
    learn_states[:,0] = sigmoid_func(learn_states[:,0],bounds[0,0],bounds[0,1])
    learn_states[:,1] = sigmoid_func(learn_states[:,1],bounds[1,0],bounds[1,1])
    learn_states[:,2] = sigmoid_func(learn_states[:,2],bounds[2,0],bounds[2,1])
    learn_states[:,5] = sigmoid_func(learn_states[:,5],bounds[3,0],bounds[3,1])
    learn_states[:,6] = sigmoid_func(learn_states[:,6],bounds[4,0],bounds[4,1])
    learn_states[:,7] = sigmoid_func(learn_states[:,7],bounds[5,0],bounds[5,1])
    return learn_states

def assimilation(T_size,delta_t,state_size,samples_size,mu0,C0,bounds,Sigma_z):
    H=np.zeros(state_size).reshape(1,-1);H[0,4]=1;H = torch.tensor(H,device=device)
    gama_num = 0.01**2;gama=torch.tensor(gama_num,device=device)
    mus = np.zeros([T_size,state_size])
    all_auxilary_learn_states2 = np.zeros([T_size,samples_size])
    learn_states,auxilary_learn_states,Ju_i_states,current_g_ui,mu0 = initialize_state(delta_t,T_size,state_size,samples_size,bounds,mu0,C0)
    mus[0,:] = mu0
    for i in range(1,T_size-1):
        t = tt[i]
        if t%5000 ==0:
            print(t)
        current_poisson=A[:,i]
        current_g_ui[:,0]=learn_states[:,0];current_g_ui[:,3]=learn_states[:,1]
        learn_states,auxilary_learn_states,Ju_i_states=\
        update_the_state(tau_ui,I_extern_Input,g_Li,g_ui,V_ui,V_L,C,V_th,V_reset,T_ref,real_g_ui,\
                         t,current_poisson,current_g_ui,learn_states,auxilary_learn_states,Ju_i_states)
        all_auxilary_learn_states2[i,:] = auxilary_learn_states[:,1]
        current_g_ui[:,0]=learn_states[:,0];current_g_ui[:,2]=learn_states[:,1]
        if t%500==0:#采样点：
            #接下来更新Ca wave
            spike_sum = all_auxilary_learn_states2[i-500:i,:].sum(axis=0)#观测区间长为500
            learn_states = update_the_state_obs(learn_states,spike_sum)
            learn_states = KF_update(learn_states,Sigma_z,bounds,state_size,gama,gama_num,H,i)
        mus[i,:]=learn_states.mean(axis=0)#采集均值
    return mus

if __name__ == '__main__':
    with open('Abos.pickle', 'rb+') as f:
        A = pickle.load(f)
    with open('Ga_xinhao_500ms_noise.pickle', 'rb+') as f:
        Yt = pickle.load(f)
        Yt = Yt.numpy()  # 观测数值
    with open('Ga_xinhao_500ms2_noise.pickle', 'rb+') as f:
        Ct = pickle.load(f)
    with open('small_cell_500ms_noise.pickle', 'rb+') as f:
        tt, uu = pickle.load(f)
    # tt为时间刻度，uu为电压
    with open('node_property.pickle', 'rb+') as f:
        node_property = pickle.load(f)

    torch.set_default_tensor_type(torch.DoubleTensor)
    if torch.cuda.is_available():
        device = torch.device("cuda")

    real_g_ui = node_property[:, 10:14]

    mus=assimilation(T_size,delta_t,state_size,samples_size,mu0,C0,bounds,Sigma_z)

    tt_500ms=[]
    time_500ms = [i*500 for i in range(T_size//500)]
    mus_500ms=[]
    for i in time_500ms:
        mus_500ms.append(mus[i,:])
        tt_500ms.append(tt[i])
    mus_500ms =np.array(mus_500ms)

    plt.plot(tt_500ms,np.array(mus_500ms)[:,0],'o')
    plt.axhline(y=0.02,ls="-",c="red")
    plt.legend(["assimilation","real gu_1"])
    plt.xlabel("t (ms)")
    plt.ylabel("gu_1")