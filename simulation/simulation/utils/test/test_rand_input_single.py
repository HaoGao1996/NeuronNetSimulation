from fishSimulation.utils import generator as gen

a = gen.rand_input_single(f=10, size=(1000, 4), num=10, ratio=(0.8, 0.5))
print(a.shape)