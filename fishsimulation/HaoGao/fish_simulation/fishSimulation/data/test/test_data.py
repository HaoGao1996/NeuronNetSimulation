from fishSimulation.data.fetcher import get_data_rd
import pickle
print(get_data_rd())

with open(get_data_rd('Abos.pickle')[-1], 'rb+') as f:
    A=pickle.load(f)
with open(get_data_rd('Ga_xinhao_500ms_noise.pickle')[-1],'rb+') as f:
    Yt=pickle.load(f)
    Yt=Yt.numpy()#观测数值
with open(get_data_rd('Ga_xinhao_500ms2_noise.pickle')[-1],'rb+') as f:
    Ct=pickle.load(f)
with open(get_data_rd('small_cell_500ms_noise.pickle')[-1],'rb+') as f:
    tt,uu=pickle.load(f)
#tt为时间刻度，uu为电压
with open(get_data_rd('node_property.pickle')[-1],'rb+') as f:
    node_property=pickle.load(f)

print(get_data_rd('Abos.pickle')[-1])