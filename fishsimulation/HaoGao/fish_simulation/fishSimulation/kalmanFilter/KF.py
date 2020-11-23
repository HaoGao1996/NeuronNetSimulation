import torch

class KalmanFilter(object):
    """
    model
    x=Fx+Bu+w
    z=Hx+v
    """
    def __init__(self, dim_x, dim_z, dim_u=0, F=None, Q=None, B=None, u=None, H=None, R=None):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # state vector
        self.x = torch.zeros(dim_x, 1)

        # state transition matrix
        if F is None:
            self.F = torch.eye(dim_x)
        else:
            self.F = F

        # process error
        if Q is None:
            self.Q = torch.eye(dim_x)
        else:
            self.Q = Q

        if dim_u == 0:
            self.B = 0
            self.u = 0
        else:
            if B is None:
                self.B = torch.zeros(dim_x, dim_u)
            else:
                self.B = B

            if u is None:
                self.u = torch.zeros(dim_u, 1)
            else:
                self.u = u

        # measurement vector
        self.z = torch.zeros(dim_z, 1)

        # measurement function
        if H is None:
            self.H = torch.zeros(dim_z, dim_x)
        else:
            self.H = H

        # measurement error
        if R is None:
            self.R = torch.eye(dim_z)
        else:
            self.R = R

        self.P = torch.eye(dim_x)                 # error covariance matrix

        self.y = torch.zeros(dim_z)          # measurement prefit residual y=z-Hx
        self.K = torch.zeros(dim_x, dim_z)               # kalman gain
        self.S = torch.zeros(dim_z, dim_z)               # system uncertainty
        self.SI = torch.zeros(dim_z, dim_z)              # inv of S
        self._I = torch.eye(dim_x)                # constant eye matrix

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()

    def predict(self):
        # x = Ax
        self.x = self.F @ self.x

        # P = APA'+Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        # save prior
        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()

    def update(self, z):
        self.z = z
        # residual between measurement and predication: y = z - Hx
        self.y = self.z - self.H @ self.x_prior

        # system uncertainty: HPH' + R
        self.S = self.H @ self.P @ self.H.T + self.R
        self.SI = torch.inverse(self.S)

        # kalman gain: K = PH'inv(S)
        self.K = self.P @ self.H.T @ self.SI

        # posterior estimate: xpost=xprior+Ky
        self.x = self.x_prior + self.K @ self.y

        # posterior covariance:
        self.P = (self._I - self.K @ self.H) @ self.P_prior

        # save posterior
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()

    def __repr__(self):
        return '\n'.join(['KalmanFilter object',
                          f'dim_x: {self.dim_x}',
                          f'dim_z: {self.dim_z}',
                          f'dim_u: {self.dim_u}'])