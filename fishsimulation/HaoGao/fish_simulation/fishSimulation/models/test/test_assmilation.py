from fishSimulation.kalmanFilter.EnKF import EnsembleKalmanFilter
from fishSimulation.models.block import Block
from fishSimulation.models.calcium import CalciumAR
from fishSimulation.utils import generator as gen

t = 1000   # experiment time
f_s = 10  # frequency of spikes
std_y = 2  # standard variance of y
f_c = 2   # frequency of camera

ca_real = gen.gen_calcium(t=t, f=f_s, std=std_y)
