import torch


class FishCalcium(object):
    def __init__(self, node_property, w_uij, delta_tb=1,
                 alpha=10, lam=(1.3, -0.5), bl=0,delta_tc=2,
                 sp_input=None, Q=None):
        """
        FishCalcium object

        :param node_property:
        :param w_uij:
            [K, N]
        :param delta_t:
        :param alpha: float
            amplitude
        :param lam: tuple
            coefficient
        :param bl: float
            baseline
        :param delta_tc: int (ms)
        :param sp_input: tensor
        :param ca_output: tensor
        :param Q: tensor
            covariance matrix of inputs
        """
        K = w_uij.shape[0]   # K: connections kind, = 4 (AMPA, NMDA, GABAa and GABAb)
        self.K = K
        N = w_uij.shape[1]   # N: numbers of neural cells
        self.N = N
        self.w_uij = w_uij   # shape: [K, N]

        self.delta_tb = delta_tb  # scalar int
        self.t = 0  # scalar int
        self.delta_tc = delta_tc

        self.update_property(node_property)

        self.t_ik_last = torch.zeros((N, ), dtype=int)  # shape [N]
        self.V_i = self.V_reset  # membrane potential, shape: [N]
        self.J_ui = torch.zeros((K, N))  # shape [K, N]

        self.counter = 0
        self.flag = False

        self.p = len(lam)
        self.cac = torch.zeros((self.p, ))          # calcium concentration
        self.alpha = torch.tensor([alpha])
        self.lam = torch.tensor(lam)
        self.bl = torch.tensor([bl])
        self.flu = self.alpha * self.bl

        self.sp_input = sp_input
        self.Q = Q

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

    def update(self):
        if self.flag is True:
            self.counter = 0
            self.flag = False

        self.t += self.delta_tb

        self.J_ui = self.J_ui * torch.exp(-self.delta_tb / self.tau_ui)
        self.J_ui += self.w_uij * self.sp_input[self.t-1]

        I_ui = self.g_ui * (self.V_ui - self.V_i) * self.J_ui
        I_syn = I_ui.sum(dim=0)

        delta_Vi = -self.g_Li * (self.V_i - self.V_L) + I_syn + self.I_extern_Input
        delta_Vi *= self.delta_tb / self.C

        Vi_normal = self.V_i + delta_Vi

        is_not_saturated = (self.t >= self.t_ik_last + self.T_ref)
        self.V_i = torch.where(is_not_saturated, Vi_normal, self.V_reset)
        active = self.V_i >= self.V_th
        self.V_i = torch.min(self.V_i, self.V_th)
        if active[0].tolist() is True:
            self.counter += 1

        self.t_ik_last = torch.where(active, torch.tensor(self.t), self.t_ik_last)

        if self.t % self.delta_tc is 0:
            cp = self.cac
            ct = torch.dot(cp, self.lam) + self.counter
            self.flag = True

            cp = cp.roll(1)
            cp[0] = ct
            self.cac = cp

            self.flu = self.alpha * (ct + self.bl)

    def enkf_predict_vi(self, x):
        """
        :param x: tensor (k-1|k-1)
            sampled xi
        :return x_: tensor (k|k-1)
            prior estimation of sampled xi
        """
        from torch.distributions.multivariate_normal import MultivariateNormal

        x_ = MultivariateNormal(loc=self.V_i, covariance_matrix=self.Q).rsample()

        return x_

    def enkf_update_vi(self, x):
        """
        :param x: tensor (k-1|k-1)
            sampled xi
        :return z_: tensor (k|k-1)
            prior estimation of sampled z_
        """

        t = self.t + self.delta_tb

        J_ui = self.J_ui * torch.exp(-self.delta_tb / self.tau_ui)
        J_ui += self.w_uij * self.sp_input[t]

        I_ui = self.g_ui * (self.V_ui - x) * J_ui
        I_syn = I_ui.sum(dim=0)

        delta_Vi = -self.g_Li * (x - self.V_L) + I_syn + self.I_extern_Input
        delta_Vi *= self.delta_tb / self.C
        x += delta_Vi

        is_not_saturated = (self.t >= self.t_ik_last + self.T_ref)
        active = x >= self.V_th
        counter = self.counter
        if active is True:
            counter += 1

        cp = self.cac
        ct = torch.dot(cp, self.lam) + counter

        cp = cp.roll(1)
        cp[0] = ct
        cac = cp

        z_ = self.alpha * (ct + self.bl)
        return z_

    def __repr__(self):
        return '/n'.join(['FishCalcium object'])