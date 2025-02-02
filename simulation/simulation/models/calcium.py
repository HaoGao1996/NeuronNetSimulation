import torch


class CalciumAR(object):
    def __init__(self, alpha=10, lam=(1.3, -0.5), bl=0, delta_t=10):
        """
        Calcium dynamics model(Autoregressive)

        :param alpha: float
            amplitude
        :param lam: tuple
            coefficient
        :param bl: float
            baseline
        :param delta_t: int (ms)
        """
        self.p = len(lam)
        self.cac = torch.zeros((self.p, ))          # calcium concentration
        self.alpha = torch.tensor([alpha])
        self.lam = torch.tensor(lam)
        self.bl = torch.tensor([bl])
        self.flu = self.alpha * self.bl

        self.delta_t = delta_t
        self.t = 0

    def update(self, s):
        """
        update fluorescence

        :param s:
            input spikes number
        :return:
            fluorescence
        """
        self.t += self.delta_t

        cp = self.cac
        ct = torch.dot(cp, self.lam) + s

        cp = cp.roll(1)
        cp[0] = ct
        self.cac = cp

        self.flu = self.alpha * (ct + self.bl)

        return self.flu

    def __repr__(self):
        return '\n'.join(['CalciumAR object'])
