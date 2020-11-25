import torch


def gen_property():
    """
    E/I: 1, bool
    blocked_in_stat: 1, bool
    I_external_Input: 1, float
    sub_blocak_idx: 1, int
    C: 1, float
    T_ref: 1, float
    g_Li: 1, float
    V_L: 1, float
    V_th: 1, float
    V_reset: 1, float
    g_ui: 4, float
    V_ui: 4, float
    tau_ui: 4, float

    total: 22
    """
    import numpy as np

    property = np.zeros((1, 22), dtype=np.float32)

    property[:, 0] = 1  # E/I
    property[:, 1] = 0  # blocked_in_stat
    property[:, 2] = 0  # I_external_Input
    property[:, 3] = 0  # sub_blocak_idx
    property[:, 4] = 1  # C
    property[:, 5] = 5  # T_ref
    property[:, 6] = 0.001  # g_Li
    property[:, 7] = -75  # V_L
    property[:, 8] = -50  # V_th
    property[:, 9] = -65  # V_reset

    property[:, 10:14] = np.array([10 / 500, 5 / 6000, 1.5 / 60, 1 / 1000])  # g_ui
    property[:, 14:18] = np.array([0, 0, -70, -100])  # V_ui
    property[:, 18:22] = np.array([2, 40, 10, 50])  # tau_ui

    return property


def gen_spikes(f):
    """

    :param f: firing rate
    :return:
    """

    s = torch.ones((10, 10000)).bernoulli_(f / 1000)
    w = torch.rand((4, 10))
    w[:2, 8:] = 0
    w[2:, :8] = 0

    return (w @ s).T.unsqueeze(dim=2)
