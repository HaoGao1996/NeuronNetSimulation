import torch
from fishSimulation.models.block import Block
from fishSimulation.models.calcium import CalciumAR

class FishCalcium(object):
    def __init__(self, node_property, w_uij, delta_tb=1,
                 alpha=10, lam=(1.3, -0.5), bl=0, delta_tc=2,
                 sp_input=None):
        """
        FishCalcium object

        _______________________________________________
        %%neural network parameters%%
        :param node_property:
        :param w_uij:
            [K, N]
        :param delta_tb: int(ms)
            lif time interval
        _______________________________________________
        %%calcium model parameters%%
        :param alpha: float
            calcium amplitude
        :param lam: tuple
            AR coefficient
        :param bl: float
            calcium baseline
        :param delta_tc: int (ms)
            calcium time interval
        _______________________________________________
        """
        self.t = 0
        self.delta_tb = delta_tb
        self.delta_tc = delta_tc

        self.block = Block(node_property=node_property, w_uij=w_uij, delta_t=delta_tb)
        self.calciumAR = CalciumAR(alpha=alpha, lam=lam, bl=bl, delta_t=delta_tc)

        self.sp_input = sp_input
        self.counter = torch.tensor([0])

    def update(self):
        self.t += self.delta_tb
        active = self.block.update(self.sp_input[self.t-1])
        self.counter = torch.where(active, self.counter+1, self.counter)

        if self.t - self.calciumAR.t == self.delta_tc:
            self.calciumAR.update(self.counter)
            self.counter = torch.tensor([0])

    def __repr__(self):
        return '/n'.join(['FishCalcium object'])
