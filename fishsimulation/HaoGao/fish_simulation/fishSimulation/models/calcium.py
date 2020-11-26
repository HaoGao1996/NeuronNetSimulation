import torch


class CalciumAR(object):
    def __init__(self, a=10, lam=(1.1, -0.15), b=1, delta_t=1):
        """
        Calcium dynamics model(Autoregressive)

        :param a: float
            amplitude
        :param lam: tuple
            coefficient
        :param b: float
            baseline
        :param delta_t: int (ms)
        """
        self.p = len(lam)
        self.c = torch.zeros((self.p, ))
        self.a = torch.tensor([a])
        self.lam = torch.tensor(lam)
        self.b = torch.tensor([b])
        self.delta_t = delta_t
        self.t = 0

    def update(self, s):
        """

        :param s:
            input spikes number
        :return:
            fluorescences
        """
        self.t += self.delta_t

        cp = self.c
        ct = torch.dot(cp, self.lam) + s

        cp = cp.roll(1)
        cp[0] = ct
        self.c = cp

        self.y = self.a * (ct + self.b)

        return self.y

    def __repr__(self):
        return '\n'.join(['CalciumAR object'])
