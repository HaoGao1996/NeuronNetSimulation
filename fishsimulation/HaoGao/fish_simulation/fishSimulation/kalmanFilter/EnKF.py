import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class EnsembleKalmanFilter(object):
    """
    model:
    x_k = f(x_k-1, u_k) + w_k
    z_k = h(x_k) + v_k
    """
    def __init__(self, dim_x, dim_z, x, z, fxu, hx, P=None, Q=None, R=None, N=100):

        self.dim_x = dim_x  # length of state vector
        self.dim_z = dim_z  # length of measurement vector

        self.x = x
        self.fxu = fxu                                              # definition of state transition function f(x)
        if Q is None:
            self.Q = torch.eye(dim_x)                                   # process uncertainty
        else:
            self.Q = Q

        self.z = torch.zeros(dim_z)                              # measurement vector
        self.hx = hx                                                # definition of observation model h(x)
        if R is None:
            self.R = torch.eye(self.dim_z)                              # measurement uncertainty
        else:
            self.R = R

        if P is None:
            self.P = torch.eye(self.dim_x)
        else:
            self.P = P  # error covariance matrix

        self.N = N                                                 # number of sampling

        self.sigmas = MultivariateNormal(loc=x, covariance_matrix=P).sample((N, ))           # samples

        self.K = torch.zeros((dim_x, dim_z))                        # kalman gain
        self.S = torch.zeros((dim_z, dim_z))                        # system uncertainty
        self.SI = torch.zeros((dim_z, dim_z))                       # inverse of system uncertainty

        # store prior estimation of x and P after predict function
        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()

        # store posterior estimation of x and P after update function
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()

        self.__mean = torch.zeros(dim_x)                            # as 1D tensor for sampling
        self.__meanz = torch.zeros(dim_z)                           # as 1D tensor for sampling

    def predict(self):
        # for i, s in enumerate(self.sigmas):
        #     self.sigmas[i] = self.fxu(s)
        self.sigmas = self.fxu(self.sigmas)

        # error for Pk|k=E(xk|k-xk|k-1)
        mn = MultivariateNormal(loc=self.__mean, covariance_matrix=self.Q)
        e = mn.sample((self.N, ))
        self.sigmas += e

        # Pk|k=1/(N-1)\sum(xk|k-xk|k-1)
        P = 0
        for s in self.sigmas:
            sx = s - self.x
            P += torch.ger(sx, sx)

        self.P = P / (self.N - 1)

        # save prior
        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()

    def update(self, z):

        # calculate system uncertainty
        sigmas_h = torch.zeros((self.N, self.dim_z))
        # for i, s in enumerate(self.sigmas):
        #     sigmas_h[i] = self.hx(s)
        sigmas_h = self.hx(sigmas_h)

        z_mean = torch.mean(sigmas_h, dim=0)

        S = torch.zeros((self.dim_z, self.dim_z))
        for sigma in sigmas_h:
            zz = sigma - z_mean  # residual
            S += torch.ger(zz, zz)
        self.S = S / (self.N - 1) + self.R
        self.SI = torch.inverse(self.S)

        # calculate cross covariance matrix
        Cxz = torch.zeros((self.dim_x, self.dim_z))
        for i in range(self.N):
            Cxz += torch.ger(self.sigmas[i]-self.x, sigmas_h[i]-z_mean)
        Cxz = Cxz / (self.N - 1)

        # kalman gain
        self.K = Cxz @ self.SI

        # sampling measurement value and update sigmas
        e = MultivariateNormal(self.__meanz, self.R).sample((self.N, ))
        for i in range(self.N):
            self.sigmas[i] += self.K @ (z + e[i] - sigmas_h[i])

        # update posterior estimation
        self.x = torch.mean(self.sigmas, dim=0)
        self.P = self.P - self.K @ self.S @ self.K.T

        # save posterior estimation
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()

    def __repr__(self):
        return '\n'.join(['EnsembleKalmanFilter object',
                          f'x: {self.x}',
                          f'P: {self.P}',
                          f'{self.sigmas.shape}'])