import torch

class KalmanFilter(object):
    def __init__(self, dim):
        self.x = torch.zeros((dim, 1))  # state vector
        self.A = torch.eye(dim)   # state transition matrix
        self.Q = torch.eye(dim)  # process error

        self.z = torch.zeros((dim, 1))    # measurement vector
        self.H = torch.zeros()  # measurement function
        self.R = torch.eye(dim)  # measurement error

        self.P = torch.eye(dim)   # error covariance matrix

        self.y = torch.zeros((dim, 1))  # residual y=z-Hx
        self.K = torch.zeros(dim)   # kalman gain
        self.PHT = torch.zeros(dim) # first component of K
        self.S = torch.zeros(dim)   # second component of K
        self.SI = torch.zeros(dim)  # inv of S
        self._I = torch.eye(dim)    # constant eye matrix

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self):

        # x = Ax
        self.x = self.A @ self.x

        # P = APA'+Q
        self.P = self.A @ self.P @ self.A.T + self.Q

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()


    def update(self):

        # residual between measurement and predication: y = z - Hx
        self.y = self.z - self.H @ self.x_prior

        # first component: PH'
        self.PHT = self.P @ self.H.T

        # second component: HPH' + R
        self.S = self.H @ self.P @ self.H.T + self.R
        self.SI = torch.inverse(self.S)

        # kalman gain: K = PH'inv(S)
        self.K = self.PHT @ self.SI

        # update estimate: xpost=xprior+Ky
        self.x = self.x_prior + self.K @ self.y

        # update covariance:
        self.P = (self._I - self.K @ self.H) @ self.P_prior

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

