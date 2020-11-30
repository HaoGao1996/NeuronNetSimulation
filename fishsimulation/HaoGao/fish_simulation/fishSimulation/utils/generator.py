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


def get_calcium(sp, alpha=10, lam=(1.1, -0.15), bl=1, delta_tc=1, std=None):
    """
    Generate calcium signal according to given spike series with a give std noise

    :param sp: torch.tensor
        spike series
    :param alpha: float
        amplitude
    :param lam: tuple
        coefficient
    :param bl: float
        baseline
    :param delta_tc
    :param std:
        add white noise with std

    :return:
        calcium dynamics
    """
    import torch
    from fishSimulation.models.calcium import CalciumAR

    t = len(sp)
    ca = CalciumAR(alpha=alpha, lam=lam, bl=bl, delta_tc=delta_tc)
    if std is None:
        return torch.tensor([ca.update(sp[i])[0].tolist() for i in range(t)])
    else:
        return torch.tensor([ca.update(sp[i])[0].tolist() for i in range(t)]) + torch.randn(t)*std


def rand_spikes(f, size):
    """
    generate random spikes

    :param f: float
        firing rate
    :param size: tuple
    :return:
        random spikes
    """
    import torch

    return torch.ones(size).bernoulli_(f / 1000)


def rand_input_single(f, size, num=10, ratio=(0.8, 0.5)):
    """
    generate input for single cell with given connections

    :param f: float
        firing rate
    :param size: tuple
        t * K
    :param num: int
        number of connections
    :param ratio: float
        E/(I+E), Ec/(Ic+Ec)

    :return:
        3D tensor
    """
    import torch

    assert len(size) == 2
    t, K = size

    s = rand_spikes(f, size=(t, num))
    w = torch.rand((num, K))

    split1 = int(ratio[0] * num)
    split2 = int(ratio[1] * K)
    w[split1:, :split2] = 0
    w[:split1, split2:] = 0

    result = torch.matmul(s, w)

    return result.unsqueeze(dim=2)


def rand_lif_spikes_single(size, f=10, delta_tb=1, num=10, ratio=(0.8, 0.5)):
    """

    :param f: float
        firing rate
    :param size: tuple
        t * K
    :param delta_tb: int(ms)
    :param num: int
        number of connections
    :param ratio: float
        E/(I+E), Ec/(Ic+Ec)
    :return:
    """
    import torch
    from fishSimulation.models.block import Block

    # t is the time
    # K is the number of connections
    t, K = size

    sp = rand_input_single(f=f, size=(t, K), num=num, ratio=ratio)
    pro = gen_property()
    b = Block(pro, w_uij=torch.ones((K,1)), delta_tb=delta_tb)

    return torch.tensor([b.update(sp[i])[0].tolist() for i in range(t)], dtype=torch.int), sp


def rand_lif_calcium_single(size, f=10, num=10, delta_tb=1, ratio=(0.8, 0.5),
                            alpha=10, lam=(1.1, -0.15), bl=1, delta_tc=1, std=None):
    """
    Generate calcium signal based on lif model which was stimulated by a random input for single cell

    :param f: float
        firing rate
    :param size: tuple
        t * K
    :param delta_tb: int(ms)
    :param num: int
        number of connections
    :param ratio: float
        E/(I+E), Ec/(Ic+Ec)
    :param alpha: float
        amplitude
    :param lam: tuple
        coefficient
    :param bl: float
        baseline
    :param delta_tc: int (ms)
    :param std: float
        noise
    :return:
        sp: spike input
        ca: calcium dynamics
    """
    import torch

    sp, sp_input = rand_lif_spikes_single(size=size, f=f, delta_tb=delta_tb, num=num, ratio=ratio)
    sampling_rate = int(delta_tc/delta_tb)
    sp0 = torch.tensor([sp[i:i+sampling_rate].sum().tolist() for i in range(0, len(sp), sampling_rate)])

    ca = get_calcium(sp=sp0, alpha=alpha, lam=lam, bl=bl, delta_tc=delta_tc, std=std)

    return ca, sp_input


# def rand_lif_input_3D(f, size, num=10, ratio=(0.8, 0.5)):
#     """
#     generate input for all cells
#
#     :param f: float
#         firing rate
#     :param size: tuple
#         t * K * N
#     :param A: int
#         number of connections
#     :param ratio: float
#         E/(I+E), Ec/(Ic+Ec)
#
#     :return:
#         3D tensor
#     """
#     import torch
#
#     assert len(size) == 3
#
#     t, K, N = size
#
#     return torch.stack([rand_lif_input_2D(f, size=(t, K), num=num, ratio=ratio) for i in range(N)], dim=2)
#
#
# def rand_lif_input_2D(f, size, num=10, ratio=(0.8, 0.5)):
#     """
#     generate input for each cell
#
#     :param f: float
#         firing rate
#     :param size: tuple
#         t * K * N
#     :param num: int
#         number of connections
#     :param ratio: float
#         E/(I+E), Ec/(Ic+Ec)
#
#     :return:
#         2D tensor
#     """
#     import torch
#
#     assert len(size) == 2
#
#     t, K = size
#
#     s = rand_spikes(f, size=(t, num))
#     w = torch.rand((num, K))
#
#     split1 = int(ratio[0] * num)
#     split2 = int(ratio[1] * K)
#     w[split1:, :split2] = 0
#     w[:split1, split2:] = 0
#
#     return torch.matmul(s, w)


#
#
# def rand_lif_calcium(size, f=10, delta_t=1, num=10, ratio=(0.8, 0.5), a=10, lam=(1.1, -0.15), b=1, std=None):
#     """
#     Generate calcium signal based on lif model which was stimulated by a random input
#
#     :return:
#         calcium dynamics
#     """
#     sp = rand_lif_spikes(size=size, f=f, delta_t=delta_t, num=num, ratio=ratio)
#
#     return get_calcium(sp=sp, a=a, lam=lam, b=b, delta_t=delta_t, std=std)



