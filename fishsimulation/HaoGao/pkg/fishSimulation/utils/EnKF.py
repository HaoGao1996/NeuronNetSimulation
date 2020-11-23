import torch
from numpy.random import multivariate_normal

class EnsembleKalmanFilter(object):
    """
    model:
    x_k = f(x_k-1, u_k) + w_k
    z_k = h(x_k) + v_k
    """
    def __init__(self, x, P, N, dim_z, fxu, hx):

        self.x = x                                                  # state vector
        self.dim_x = len(self.x)                                    # length of state vector
        self.Q = torch.eye(self.dim_x)                              # process uncertainty
        self.z = torch.zeros(dim_z)                                 # measurement vector
        self.dim_z = dim_z                                          # length of measurement vector
        self.R = torch.eye(self.dim_z)                              # measurement uncertainty

        self.P = P                                                  # error covariance matrix
        self.N = N                                                  # number of sampling
        self.sigmas = multivariate_normal(mean=x, cov=P, size=N)    # samples
        self.fxu = fxu                                              # definition of state transition function f(x)
        self.hx = hx                                                # definition of observation model h(x)

        self.K = torch.zeros((self.dim_x, self.dim_z))              # kalman gain
        self.S = torch.zeros((self.dim_z, self.dim_z))              # system uncertainty
        self.SI = torch.zeros((self.dim_z, self.dim_z))             # inverse of system uncertainty

        # store prior estimation of x and P after predict function
        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()

        # store posterior estimation of x and P after update function
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()

        self._mean = torch.zeros(self.dim_x)

    def predict(self):
        for i, s in enumerate(self.sigmas):
            self.sigmas[i] = self.fxu(s)

        self.sigmas += multivariate_normal(mean=self._mean, cov=self.Q, size=self.N)

        P = 0
        for s in self.sigmas:
            sx = s - self.x
            P += torch.dot(sx, sx)

        self.P = P / (self.N - 1)

        # save prior
        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()

    def update(self, z, R):
        sigmas_h = torch.zeros((self.N, self.dim_z))
        for i, s in enumerate(self.sigmas):
            sigmas_h[i] = self.hx(s)

        z_mean = torch.mean(sigmas_h, axis=0)



    def __repr__(self):
        return '\n'.join(['EnsembleKalmanFilter object',
                          f'x: {self.x}',
                          f'P: {self.P}',
                          f'{self.sigmas.shape}'])