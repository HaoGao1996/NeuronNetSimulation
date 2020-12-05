import torch
from fishSimulation.models.block import Block
from fishSimulation.models.calcium import CalciumAR
from fishSimulation.models.fishCalcium import FishCalcium
from torch.distributions.multivariate_normal import MultivariateNormal
import copy


class FishCalciumEnKF(object):
    def __init__(self, node_property, w_uij, delta_tb=1,
                 alpha=10, lam=(1.3, -0.5), bl=0, delta_tc=2,
                 N=100, P=None, Q=None, R=None,
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
        %%EnKF parameters%%
        :param N:
            sampling number
        :param P: tensor
            error covariance matrix
        :param Q: tensor
            process covariance matrix
        :param R: tensor
            measurement covariance matrix
        _______________________________________________
        """
        self.fc = FishCalcium(node_property=node_property, w_uij=w_uij, delta_tb=delta_tb,
                              alpha=alpha, lam=lam, bl=bl, delta_tc=delta_tc,
                              sp_input=sp_input)

        self.t_ratio = int(delta_tc / delta_tb)

        self.x = self.obtain_vars(self.fc)
        self.dim_x = self.x.shape[0]
        self.dim_z = self.fc.calciumAR.flu.shape[0]

        self.N = N

        if P is None:
            self.P = torch.eye(self.dim_x)                             # error covariance matrix
        else:
            self.P = P

        if Q is None:
            self.Q = torch.eye(self.dim_x)                               # process uncertainty
        else:
            self.Q = Q

        if R is None:
            self.R = torch.eye(self.dim_z)                             # measurement uncertainty
        else:
            self.R = R

        self.sp_input = sp_input

        self.K = torch.zeros((self.dim_x, self.dim_z))                        # kalman gain
        self.S = torch.zeros((self.dim_z, self.dim_z))                        # system uncertainty
        self.SI = torch.zeros((self.dim_z, self.dim_z))                       # inverse of system uncertainty

        self.__mean = torch.zeros(self.dim_x)                            # as 1D tensor for sampling
        self.__meanz = torch.zeros(self.dim_z)                           # as 1D tensor for sampling

    def init_sampling(self):

        self.x_samples = MultivariateNormal(loc=self.x, covariance_matrix=self.P).sample((self.N,))
        # initialize N block objects
        self.fc_list = [self.update_vars(copy.deepcopy(self.fc), self.x_samples[i]) for i in range(self.N)]

    # def bound_vars(self):
    #     assert a < b
    #     return (a * torch.exp(-x) + b) / (1 + torch.exp(-x))

    def obtain_vars(self, fc):
        x = torch.cat((fc.block.g_ui[0], fc.block.g_ui[2]), dim=0)
        return x

    def update_vars(self, fc, x):
        fc.block.g_ui[0], fc.block.g_ui[2] = x
        return fc

    def cal_x_mean(self):
        x_all = torch.stack([self.obtain_vars(self.fc_list[i]) for i in range(self.N)], dim=0)
        x_mean = x_all.mean(0)

        return x_mean

    def cal_vi_mean(self):
        vi_all = torch.stack([self.fc_list[i].block.V_i for i in range(self.N)], dim=0)
        vi_mean = vi_all.mean(0)

        return vi_mean

    def run_enkf(self, z):
        # predict step
        for i in range(self.N):
            for _ in range(self.t_ratio):
                self.fc_list[i].update()
        self.x_samples = torch.stack([self.obtain_vars(self.fc_list[i]) for i in range(self.N)], dim=0)

        P = cal_cov(self.x_samples) + self.Q

        x_samples_ca = torch.stack([self.fc_list[i].calciumAR.flu for i in range(self.N)], dim=0)
        self.S = cal_cov(x_samples_ca) + self.R
        self.SI = torch.inverse(self.S)

        # calculate cross covariance matrix
        Cxz = cal_cross_cov(self.x_samples, x_samples_ca)

        # kalman gain
        self.K = Cxz @ self.SI

        # sampling measurement value and update sigmas
        e = MultivariateNormal(self.__meanz, self.R).sample((self.N, ))
        for i in range(self.N):
            self.sigmas[i] += self.K @ (z + e[i] - sigmas_h[i])

        # update posterior estimation
        self.x = torch.mean(self.sigmas, dim=0)
        self.P = self.P - self.K @ self.S @ self.K.T

    def __repr__(self):
        return '/n'.join(['FishCalciumEnKF object'])


def cal_cov(x):
    n = x.shape[0]
    x_mean = x.mean(0)
    cov = 0
    for i in range(n):
        sx = x[i] - x_mean
        cov += torch.ger(sx, sx)

    cov = cov / (n - 1)
    return cov


def cal_cross_cov(x, z):
    cov = 0
    n = x.shape[0]
    x_mean = x.mean(0)
    z_mean = z.mean(0)

    for i in range(n):
        cov += torch.ger(x[i] - x_mean, z[i] - z_mean)
    cov = cov / (n - 1)

    return cov
