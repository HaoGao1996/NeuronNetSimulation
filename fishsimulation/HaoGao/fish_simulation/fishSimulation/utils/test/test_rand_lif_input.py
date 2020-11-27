from fishSimulation.utils import generator as gen

a2d = gen.rand_lif_input_2D(f=10, size=(1000, 4), num=10, ratio=(0.8, 0.5))
print(a2d.shape)

a3d2 = gen.rand_lif_input_3D(f=10, size=(1000, 4, 2), num=10, ratio=(0.8, 0.5))
print(a3d2.shape)

a3d1 = gen.rand_lif_input_3D(f=10, size=(1000, 4, 1), num=10, ratio=(0.8, 0.5))
print(a3d1.shape)

print('finished')
