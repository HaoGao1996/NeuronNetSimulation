from fishSimulation.models.block import Block
from fishSimulation.models.calcium import CalciumAR


class FishCalcium(Block, CalciumAR):
    def __init__(self, node_property, w_uij, alpha=10, lam=(1.3, -0.5), bl=0, delta_tb=1, delta_tc=2):
        Block.__init__(self, node_property=node_property, w_uij=w_uij, delta_tb=delta_tb)
        CalciumAR.__init__(self, alpha=alpha, lam=lam, bl=bl, delta_tc=delta_tc)

    def update_block(self, d):
        Block.update(self, d=d)

    def update_calcium(self, s):
        CalciumAR.update(self, s=s)

    def __repr__(self):
        return '/n'.join(['FishCalcium object'])

