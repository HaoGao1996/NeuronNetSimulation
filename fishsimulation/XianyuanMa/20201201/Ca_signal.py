import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('80000Vi.pickle','rb+') as f:
    V_i=pickle.load(f)[0]

T_size = V_i.shape[0];N = V_i.shape[1]
np.random.seed(42)
mu=0
sigma=0.01**2
epslo= np.random.normal(mu, sigma, [T_size,N])#噪音

#将电压转化为spike信号
s2=np.zeros([T_size,N])
obs_interval = 200
for i in range(1,T_size):
    if i%obs_interval==0:
        V_i_ = V_i[i-obs_interval:i,:]
        s2[i,:]=(V_i_>=-50).sum(axis=0)

yj=0.2 #equation 1中的参数
a=0.3;b=0.4#equation 2中的参数
#初值设置在a*b+噪音
#Ca signal generator
Ct=torch.zeros(T_size,N)
Yt=torch.zeros(T_size,N)
for i in range(T_size-1):
    Ct[i+1,:]=yj*Ct[i,:]+s2[i+1,:]
for i in range(T_size-1):
    Yt[i+1,:]=a*(Ct[i,:]+b)+epslo[i,:]

with open('80000_Ca_signal_500ms.pickle','wb') as f:
    pickle.dump(Yt,f)
with open('80000_Ca_signal_500ms2.pickle','wb') as f:
    pickle.dump(Ct,f)
with open('80000_Ca_signal_property.pickle','wb') as f:
    pickle.dump([a,b,yj],f)