import torch


class Block(object):
    def __init__(self, node_property, w_uij=None, delta_t=1):
        """
        A block is a set of spiking neurons with inner full connections, we consider 4 type connections:
        AMPA, NMDA, GABAa and GABAb


        :param node_property:
        :param delta_t:
        """

        if w_uij is None:
            N = 1  # N: numbers of neural cells
            K = 4  # K: connections kind, = 4 (AMPA, NMDA, GABAa and GABAb)
        else:
            N = w_uij.shape[1]
            K = w_uij.shape[0]

        self.delta_t = delta_t

        self.update_property(node_property)

        self.t_ik_last = torch.zeros((N, ))  # shape [N]
        self.active = torch.tensor([False])  # bool
        self.V_i = torch.ones((N, )) * (self.V_th + self.V_reset) / 2  # membrane potential, shape: [N]
        self.J_ui = torch.zeros((K, N))  # shape [K, N]

        self.t = torch.tensor([0.])  # scalar



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

    def update(self, poisson):
        """

        :param poisson: input
        :return:
        """
        self.t += self.delta_t

        self.J_ui = self.J_ui * torch.exp(-self.delta_t / self.tau_ui)
        self.J_ui += poisson

        I_ui = self.g_ui * (self.V_ui - self.V_i) * self.J_ui
        I_syn = I_ui.sum(dim=0)

        delta_Vi = -self.g_Li * (self.V_i - self.V_L) + I_syn + self.I_extern_Input
        delta_Vi *= self.delta_t / self.C

        Vi_normal = self.V_i + delta_Vi

        is_not_saturated = (self.t >= self.t_ik_last + self.T_ref)
        self.V_i = torch.where(is_not_saturated, Vi_normal, self.V_reset)
        self.active = self.V_i >= self.V_th
        self.V_i = torch.min(self.V_i, self.V_th)

        self.t_ik_last = torch.where(self.active, self.t, self.t_ik_last)

        return self.active

    def __repr__(self):
        return '\n'.join(['block object'])



