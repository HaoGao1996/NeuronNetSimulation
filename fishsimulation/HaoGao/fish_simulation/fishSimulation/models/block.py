import torch


class Block(object):
    def __init__(self, node_property, w_uij, delta_tb=1):
        """
        A block is a set of spiking neurons with inner full connections, we consider 4 type connections:
        AMPA, NMDA, GABAa and GABAb

        :param node_property:
        :param w_uij:
            [K, N]
        :param delta_t:
        """

        K = w_uij.shape[0]   # K: connections kind, = 4 (AMPA, NMDA, GABAa and GABAb)
        self.K = K
        N = w_uij.shape[1]   # N: numbers of neural cells
        self.N = N
        self.w_uij = w_uij   # shape: [K, N]

        self.delta_tb = delta_tb  # scalar int
        self.t = 0  # scalar int

        self.update_property(node_property)

        self.t_ik_last = torch.zeros((N, ), dtype=int)  # shape [N]
        self.V_i = torch.ones((N, )) * (self.V_th + self.V_reset) / 2  # membrane potential, shape: [N]
        self.J_ui = torch.zeros((K, N))  # shape [K, N]

        self.counter = 0

    def update_property(self, node_property):
        self.I_extern_Input = torch.tensor(node_property[:, 2])  # extern_input index , shape[K]
        self.sub_idx = torch.tensor(node_property[:, 3])  # shape [N]
        self.C = torch.tensor(node_property[:, 4])  # shape [N]
        self.T_ref = torch.tensor(node_property[:, 5])  # shape [N]
        self.g_Li = torch.tensor(node_property[:, 6])  # shape [N]
        self.V_L = torch.tensor(node_property[:, 7])  # shape [N]
        self.V_th = torch.tensor(node_property[:, 8])  # shape [N]
        self.V_reset = torch.tensor(node_property[:, 9])  # shape [N]
        self.g_ui = torch.tensor(node_property[:, 10:14].reshape([4, 1]))  # shape [K, N]
        self.V_ui = torch.tensor(
            node_property[:, 14:18].reshape([4, 1]))  # AMPA, NMDA, GABAa and GABAb potential, shape [K, N]
        self.tau_ui = torch.tensor(node_property[:, 18:22].reshape([4, 1]))  # shape [K, N]

    def update(self, d):
        """
        :param d: tensor
            density of each connection: shape [K, N]
        :return:
            active, shape [N, 1]
        """
        self.t += self.delta_tb

        self.J_ui = self.J_ui * torch.exp(-self.delta_tb / self.tau_ui)
        self.J_ui += self.w_uij * d

        I_ui = self.g_ui * (self.V_ui - self.V_i) * self.J_ui
        I_syn = I_ui.sum(dim=0)

        delta_Vi = -self.g_Li * (self.V_i - self.V_L) + I_syn + self.I_extern_Input
        delta_Vi *= self.delta_tb / self.C

        Vi_normal = self.V_i + delta_Vi

        is_not_saturated = (self.t >= self.t_ik_last + self.T_ref)
        self.V_i = torch.where(is_not_saturated, Vi_normal, self.V_reset)
        active = self.V_i >= self.V_th
        self.V_i = torch.min(self.V_i, self.V_th)
        if active is True:
            self.counter += 1

        self.t_ik_last = torch.where(active, torch.tensor(self.t), self.t_ik_last)

        return active

    def __repr__(self):
        return '\n'.join(['Block object'])



