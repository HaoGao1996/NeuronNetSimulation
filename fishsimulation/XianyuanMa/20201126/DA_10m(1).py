import sys
sys.path.append("/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/")
sys.path.append("/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/cuda")
sys.path.append("/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/cuda/python")
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import torch
# from brain_block.bold_model_pytorch import BOLD
from brain_block.bold_model import BOLD
import numpy as np
import copy
import time
import os
from cuda.python.dist_blockwrapper import BlockWrapper as block_gpu
from multiprocessing.pool import ThreadPool as pool
from scipy.io import loadmat
import pandas as pd
import matplotlib
import matplotlib.pyplot as mp


def ge_brain_index(properties):
    brain_index = copy.deepcopy(properties[:, 3])
    brain_index = np.array(brain_index).astype(np.int)
    nod_name, nod_sum = np.unique(brain_index, return_counts=True)
    return brain_index, nod_name, nod_sum


def ge_parameter(hp_range, para_ind, ensembles, brain_num, brain_index):
    # hp_range = np.array([[1,2],[3,4],[5,6],[7,8]])
    # para_ind = np.array([10,11,12,13], dtype=int)
    hp_num = len(para_ind)
    hp_low = np.tile(hp_range[para_ind - 10, 0], (brain_num, 1))  # shape = brain_num*hp_num
    hp_high = np.tile(hp_range[para_ind - 10, 1], (brain_num, 1))
    hp = np.linspace(hp_low, hp_high, 3*ensembles)[ensembles:-1*ensembles]  # shape = ensembles, brain_num, hp_num
    for i in range(hp_num):
        idx = np.random.choice(ensembles, ensembles, replace=False)
        hp[:, :, i] = hp[idx, :, i]
    para = np.random.exponential(np.ones([ensembles, len(brain_index), hp_num]))
    pip = np.zeros([len(brain_index)*ensembles*hp_num, 2])
    pip[:, 0] = np.repeat(np.arange(len(brain_index)*ensembles), hp_num)
    pip[:, 1] = np.tile(para_ind, len(brain_index)*ensembles)
    hpip = np.zeros([brain_num*ensembles*hp_num, 2])
    hpip[:, 0] = np.repeat(np.arange(brain_num*ensembles), hp_num)
    hpip[:, 1] = np.tile(para_ind, brain_num*ensembles)
    return hp_num, log_abs(hp, hp_low, hp_high), para, hp_low, hp_high, pip.astype(np.uint32), hpip.astype(np.uint32)


def log_abs(val, lower, upper, scale=10):
    if (val <= upper).all() and (val >= lower).all():
        return scale * (np.log(val - lower) - np.log(upper - val))
    else:
        return None


def sigmoid_abs(val, lower, upper, scale=10):
    assert np.isfinite(val).all()
    return lower + (upper - lower) / (1 + np.exp(-val / scale))


def ensemble_system(Block, Bold, steps, w, hp_fore, para, hp_num, brain_n, ensembles, k, hp_sigma, hp_low, hp_high, brain_index, nod_sum, bold_sigma, pip, hpip):
    hp_transf_enkf = w[:, :brain_n*hp_num].reshape(ensembles, brain_n, hp_num) + np.sqrt(hp_sigma) * np.random.randn(ensembles, brain_n, hp_num)
    hp_enkf = sigmoid_abs(hp_transf_enkf, hp_low, hp_high)
    hp_delta = hp_enkf/sigmoid_abs(hp_fore.reshape(ensembles, brain_n, hp_num), hp_low, hp_high)
    # Block.mul_property_by_subblk(hpip, hp_delta.reshape(-1).astype(np.float32), accumulate=True)
    # Block.update_property(pip, (para*hp_enkf[:, brain_index, :]).reshape(ensembles * k * hp_num).astype(np.float32))
    Block.mul_property_by_subblk(hpip, hp_enkf.reshape(-1).astype(np.float32))

    act = Block.run(steps, freqs=True, vmean=False, sample_for_show=False)
    act = act/np.tile(nod_sum, ensembles)
    print(np.array(act).max(), np.array(act).min(), np.array(act).mean())

    df = Block.last_time_stat()
    for t in range(steps):
        bold = Bold.run(np.maximum(act[t], 1e-05))
    bold = bold + np.sqrt(bold_sigma) * np.random.randn(5, ensembles*brain_n)
    w_hat = np.concatenate((hp_transf_enkf.reshape(ensembles, brain_n*hp_num), (act[-1]).reshape([ensembles, brain_n]),
                            bold[0].reshape([ensembles, brain_n]), bold[1].reshape([ensembles, brain_n]),
                            bold[2].reshape([ensembles, brain_n]), bold[3].reshape([ensembles, brain_n]),
                            bold[4].reshape([ensembles, brain_n])), axis=1)
    return w_hat, hp_transf_enkf, df, act


def distributed_kalman(w_hat, brain_n, ensembles, bold_sigma, bold_y_t, rate, hp_num):
    w = w_hat[:, :brain_n*(hp_num+6)].copy()
    w_mean = np.mean(w, axis=0, keepdims=True)
    w_diff = w - w_mean
    w_cx = w_diff[:, -brain_n:] * w_diff[:, -brain_n:]
    w_cxx = np.sum(w_cx, axis=0) / (ensembles - 1) + bold_sigma
    kalman = np.dot(w_diff[:, -brain_n:].T, w_diff) / (w_cxx.reshape([brain_n, 1])) / (ensembles - 1) # (brain_n, w_shape[1])
    w_ensemble = w + (bold_y_t[0, :, None] + np.sqrt(bold_sigma) * np.random.randn(brain_n, ensembles)
                      - w[:, -brain_n:].T)[:, :, None] * kalman[:, None, :]  # (brain_n, ensembles, w_shape[1])
    w_hat[:, brain_n*hp_num:brain_n*(hp_num+6)] = rate * w_ensemble[:, :, -6*brain_n:].reshape(
                                        [brain_n, ensembles, 6, brain_n]).diagonal(0, 0, 3).reshape([-1, 6 * brain_n])
    w_hat[:, :brain_n * hp_num] = rate * w_ensemble[:, :, :brain_n * hp_num].reshape(
             [brain_n, ensembles, brain_n, hp_num]).diagonal(0, 0, 2).transpose(0, 2, 1).reshape([-1, brain_n * hp_num])
    w_hat[:, :brain_n*(hp_num+6)] = w_hat[:, :brain_n*(hp_num+6)] + (1 - rate) * np.mean(w_ensemble, axis=0)
    print(w_hat[:, :brain_n*(hp_num+6)].max(), w_hat[:, :brain_n*(hp_num+6)].min())
    return w_hat


def da_show(W, data, T, path, brain_num):
    iteration = [i for i in range(T)]
    for i in range(brain_num):
        print("show"+str(i))
        fig = plt.figure(figsize=(8, 4), dpi=500)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(iteration, data[:T, i], 'r-')
        ax1.plot(iteration, np.mean(W[:T, :, -brain_num+i], axis=1), 'b-')
        plt.fill_between(iteration, np.mean(W[:T, :, -brain_num+i], axis=1) -
                         np.sqrt(np.var(W[:T, :, -brain_num+i], axis=1)), np.mean(W[:T, :, -brain_num+i], axis=1)
                         + np.sqrt(np.var(W[:T, :, -brain_num+i], axis=1)), color='b', alpha=0.2)
        plt.ylim((0.0, 0.08))
        plt.savefig(os.path.join(path, "bold"+str(i)+".png"))
        plt.close(fig)


def show_hp(w, hp_low, hp_high, T, path, brain_num, parameter_index_in_property):
    iteration = [i for i in range(T)]
    hp_num = len(parameter_index_in_property)
    w_ = w[:, :, :brain_num*hp_num].copy().reshape([w.shape[0]*w.shape[1], brain_num, hp_num])
    W = sigmoid_abs(w_, hp_low, hp_high).reshape([w.shape[0], w.shape[1], brain_num*hp_num])
    for i in range(brain_num*hp_num):
        print("show"+str(i))
        fig = plt.figure(figsize=(8, 4), dpi=500)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(iteration, np.mean(W[:T, :, i], axis=1), 'b-')
        plt.fill_between(iteration, np.mean(W[:T, :, i], axis=1) -
                         np.sqrt(np.var(W[:T, :, i], axis=1)), np.mean(W[:T, :, i], axis=1)
                         + np.sqrt(np.var(W[:T, :, i], axis=1)), color='b', alpha=0.2)
        plt.savefig(os.path.join(path, "hp"+str(i)+".png"))
        plt.close(fig)


def transfer_to_relative(path, bid, output_path=None):
    if output_path is None:
        output_path = path
    path = os.path.join(path, "block_{}.npz".format(bid))
    output_path = os.path.join(output_path, "block_{}".format(bid))

    file = np.load(path)
    file = {key: file[key] for key in file.keys()}
    
    file['input_block_idx'] = np.ascontiguousarray(file['input_block_idx'].astype(np.int16))
    file['input_block_idx'] -= bid
    
    file['property'] = np.ascontiguousarray(file['property'].astype(np.float32))
    file['property'][:, 3] -= 5400
    np.savez(output_path, **file)


def main():
    start = time.time()
    ensembles = 60
    # k = 10004232
    # k = 10001904
    hp_sigma = 1
    bold_sigma = 1e-6

    degree = 100
    para_ind = np.array([10, 12], dtype=int)
    epsilon = 200
    hp_range_rate = np.array([3, 3])
    distributed_rate = 0.5

    # block_path = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/block_all"
    block_path = "./single10m/block_92"
    path = "./single10m/92_degree"+str(degree)+str(para_ind)+str(epsilon)+str(hp_range_rate)+str(distributed_rate)+str(ensembles)
    os.makedirs(path, exist_ok=True)
    T = 450
    brain_num = 92
    noise_rate = 0.01
    steps = 800

    bold_y = loadmat("./single10m/AAL_rest_LGN.mat")
    bold_y = bold_y['rest_timecourse']
    bold_y = np.array(bold_y)[:, :92]
    bold_y = 0.03 + 0.05 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())

    arr = np.load("./single10m/single/block_0.npz")

    property = arr["property"]
    k = len(property)
    property = arr["property"][:k]

    brain_index, nod_name, nod_sum = ge_brain_index(property)
    hp_real = property[0, 10:14].reshape([-1, 1])
    hp_range = np.concatenate((hp_real / hp_range_rate[0], hp_real * hp_range_rate[1]), axis=1)
    hp_num, hp, para, hp_low, hp_high, pip, hpip = ge_parameter(hp_range, para_ind, ensembles, brain_num, brain_index)
    print("ge_parameter:"+str(time.time()-start))

    Block = block_gpu('10.5.4.1:50051', block_path, noise_rate, 1., True)
    print("ge_Block:"+str(time.time()-start))

    for i in range(ensembles):
        Block.update_property(pip[i*k*hp_num:(i+1)*k*hp_num], para.reshape(ensembles * k * hp_num)[i*k*hp_num:(i+1)*k*hp_num].astype(np.float32))
    Block.mul_property_by_subblk(hpip, sigmoid_abs(hp, hp_low, hp_high).reshape(-1).astype(np.float32))
    print("update_Block:"+str(time.time()-start))

    Bold = BOLD(epsilon=epsilon, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)

    act = Block.run(steps, freqs=True, vmean=False, sample_for_show=False)

    print("run block:"+str(time.time()-start))
    for t in range(steps):
        bold = Bold.run(np.maximum(act[t]/np.tile(nod_sum, ensembles), 1e-05))
    print("run bold:" + str(time.time() - start))
    w = np.concatenate((hp.reshape([ensembles, brain_num*hp_num]),
                        (act[-1]/np.tile(nod_sum, ensembles)).reshape([ensembles, brain_num]),
                        bold[0].reshape([ensembles, brain_num]), bold[1].reshape([ensembles, brain_num]),
                        bold[2].reshape([ensembles, brain_num]), bold[3].reshape([ensembles, brain_num]),
                        bold[4].reshape([ensembles, brain_num])), axis=1)
    w_save = [w[:, :brain_num * (hp_num + 6)]]
    hp_fore = hp.copy().reshape([ensembles, brain_num * hp_num])
    print("begin da:" + str(time.time() - start))

    for t in range(T - 1):
        start = time.time()
        bold_y_t = bold_y[t].reshape([1, brain_num])
        w_hat, hp_fore, df, act = ensemble_system(Block, Bold, steps, w, hp_fore, para, hp_num, brain_num, ensembles, k,
                                                  hp_sigma, hp_low, hp_high, brain_index, nod_sum, bold_sigma, pip,
                                                  hpip)
        print("run da_es:" + str(time.time() - start))
        w_save.append(w_hat[:, :brain_num * (hp_num + 6)].copy())
        if t <= 28 or t % 30 == 28 or t == (T - 2):
            np.save(os.path.join(path, "W" + ".npy"), w_save)
            # df.to_csv(os.path.join(path, "data" + str(t) + ".csv"))
            # np.save(os.path.join(path, "act" + str(t) + ".npy"), act)
        w = distributed_kalman(w_hat, brain_num, ensembles, bold_sigma, bold_y_t, distributed_rate, hp_num)
        print("run da_dk:" + str(time.time() - start))
        """
        Bold.s = torch.from_numpy(
            w[:, (hp_num + 1) * brain_num: (hp_num + 2) * brain_num].copy().reshape([ensembles * brain_num])).cuda()
        Bold.q = torch.from_numpy(np.maximum(w[:, (hp_num + 2) * brain_num: (hp_num + 3) * brain_num], 1e-05).reshape(
            [ensembles * brain_num])).cuda()
        Bold.v = torch.from_numpy(np.maximum(w[:, (hp_num + 3) * brain_num: (hp_num + 4) * brain_num], 1e-05).reshape(
            [ensembles * brain_num])).cuda()
        Bold.log_f_in = torch.from_numpy(
            np.maximum(w[:, (hp_num + 4) * brain_num: (hp_num + 5) * brain_num], -15).reshape(
                [ensembles * brain_num])).cuda()
        """
        Bold.s = w[:, (hp_num + 1) * brain_num: (hp_num + 2) * brain_num].copy().reshape([ensembles * brain_num])
        Bold.q = np.maximum(w[:, (hp_num + 2) * brain_num: (hp_num + 3) * brain_num], 1e-05).reshape(
            [ensembles * brain_num])
        Bold.v = np.maximum(w[:, (hp_num + 3) * brain_num: (hp_num + 4) * brain_num], 1e-05).reshape(
            [ensembles * brain_num])
        Bold.f_in = np.maximum(w[:, (hp_num + 4) * brain_num: (hp_num + 5) * brain_num], -15).reshape(
                [ensembles * brain_num])
        print("------------run da" + str(t) + ":" + str(time.time() - start))
    da_show(np.array(w_save), bold_y, T, path, brain_num)
    show_hp(np.array(w_save), hp_low, hp_high, T, path, brain_num, para_ind)
    Block.shutdown()


def load_if_exist(func, *args):
    path = os.path.join(*args)
    if os.path.exists(path + ".npy"):
        out = np.load(path + ".npy")
    else:
        out = func()
        np.save(path, out)
    return out


def block_divide(alpha, beta):
    path_in = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/v100"
    path_out = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/block_" + str(alpha) + str(beta)
    os.makedirs(path_out, exist_ok=True)

    file = np.load(os.path.join(path_in, 'block_0.npz'))

    property = file["property"]
    output_neuron_idx = file["output_neuron_idx"]
    input_block_idx = file["input_block_idx"]
    input_neuron_idx = file["input_neuron_idx"]
    input_channel_offset = file["input_channel_offset"]
    weight = file["weight"]
    """
    idx = (input_neuron_idx % 2 == output_neuron_idx % 2)
    output_neuron_idx = np.ascontiguousarray(output_neuron_idx[idx].astype(np.uint32))
    input_block_idx = np.ascontiguousarray(input_block_idx[idx].astype(np.int16))
    input_neuron_idx = np.ascontiguousarray(input_neuron_idx[idx].astype(np.uint32))
    input_channel_offset = np.ascontiguousarray(input_channel_offset[idx].astype(np.uint8))
    weight = np.ascontiguousarray(weight[idx])
    """

    idxalpha = (input_neuron_idx % 2 == 0) * (output_neuron_idx % 2 == 1)
    idxbeta = (input_neuron_idx % 2 == 1) * (output_neuron_idx % 2 == 0)
    weight[idxalpha] = weight[idxalpha] * alpha
    weight[idxbeta] = weight[idxbeta] * beta

    np.savez(os.path.join(path_out, 'block_0.npz'), property=property,
             output_neuron_idx=output_neuron_idx,
             input_block_idx=input_block_idx,
             input_neuron_idx=input_neuron_idx,
             input_channel_offset=input_channel_offset,
             weight=weight)


def block_v1_divide():
    path_in = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/v100"
    file = np.load(os.path.join(path_in, 'block_0.npz'))

    property = file["property"]
    output_neuron_idx = file["output_neuron_idx"]
    input_block_idx = file["input_block_idx"]
    input_neuron_idx = file["input_neuron_idx"]
    input_channel_offset = file["input_channel_offset"]
    weight = file["weight"]
    '''
    path_out = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/block_v1lgn_solo"
    idx_out = np.isin(property[output_neuron_idx, 3], np.array([42, 43, 90, 91, 134, 135, 182, 183], dtype=output_neuron_idx.dtype))
    idx_in = 1 - np.isin(property[input_neuron_idx, 3], np.array([42, 43, 90, 91, 134, 135, 182, 183], dtype=input_neuron_idx.dtype))
    idx_block = np.isin(input_block_idx, np.array([0]))
    idx = (1 - np.logical_and(idx_block, np.logical_and(idx_out, idx_in))).nonzero()[0]
    ''''''
    path_out = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/block_v1lgn_solo90"
    idx_out = np.isin(property[output_neuron_idx, 3], np.array([42, 43, 134, 135], dtype=output_neuron_idx.dtype))
    idx_in = 1 - np.isin(property[input_neuron_idx, 3], np.array([42, 43, 134, 135], dtype=input_neuron_idx.dtype))
    idx_block = np.isin(input_block_idx, np.array([0]))
    idx = (1 - np.logical_and(idx_block, np.logical_and(idx_out, idx_in))).nonzero()[0]
    '''
    path_out = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/v10090"
    idx_out = np.isin(property[output_neuron_idx, 3], np.array([90, 91, 182, 183], dtype=output_neuron_idx.dtype))
    idx_in = np.isin(property[input_neuron_idx, 3], np.array([90, 91, 182, 183], dtype=input_neuron_idx.dtype))
    idx_block = np.isin(input_block_idx, np.array([0]))
    idx = (1 - np.logical_and(idx_block, np.logical_or(idx_out, idx_in))).nonzero()[0]

    output_neuron_idx = np.ascontiguousarray(output_neuron_idx[idx].astype(np.uint32))
    input_block_idx = np.ascontiguousarray(input_block_idx[idx].astype(np.int16))
    input_neuron_idx = np.ascontiguousarray(input_neuron_idx[idx].astype(np.uint32))
    input_channel_offset = np.ascontiguousarray(input_channel_offset[idx].astype(np.uint8))
    weight = np.ascontiguousarray(weight[idx])

    os.makedirs(path_out, exist_ok=True)
    np.savez(os.path.join(path_out, 'block_0.npz'), property=property,
             output_neuron_idx=output_neuron_idx,
             input_block_idx=input_block_idx,
             input_neuron_idx=input_neuron_idx,
             input_channel_offset=input_channel_offset,
             weight=weight)


def simulation_draw(W, data, T, path, brain_num):
    iteration = [i for i in range(T)]
    path = path + '/show'
    os.makedirs(path, exist_ok=True)
    for i in range(brain_num):
        print("show"+str(i))
        fig = plt.figure(figsize=(8, 4), dpi=500)
        ax1 = fig.add_subplot(1, 1, 1)
        # ax1.plot(iteration, data[:T, 0, i], 'b-', label='changed states')
        ax1.plot(iteration, data[:T, 1, i], 'g-', label='unchanged states')
        ax1.plot(iteration, np.mean(W[:T, :, -brain_num+i], axis=1), 'r-', label='before simulation')
        plt.fill_between(iteration, np.mean(W[:T, :, -brain_num+i], axis=1) -
                         np.sqrt(np.var(W[:T, :, -brain_num+i], axis=1)), np.mean(W[:T, :, -brain_num+i], axis=1)
                         + np.sqrt(np.var(W[:T, :, -brain_num+i], axis=1)), color='b', alpha=0.2)
        plt.legend()
        plt.ylim((0.0, 0.1))
        plt.savefig(os.path.join(path, "bold"+str(i)+".png"))
        plt.close(fig)


def draw_pic(data1, data2, T, path, brain_num):
    iteration = [i for i in range(T)]
    path = path + '/compare'
    os.makedirs(path, exist_ok=True)
    for i in range(brain_num):
        print("show"+str(i))
        fig = plt.figure(figsize=(8, 4), dpi=500)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(iteration, data1[:T, 1, i], 'r-', label='line 1')
        ax1.plot(iteration, data2[:T, 1, i], 'b-', label='line2')
        plt.legend()
        # plt.ylim((0.0, 0.1))
        plt.savefig(os.path.join(path, "bold"+str(i)+".png"))
        plt.close(fig)


def other_para(para_ind, shift_rate, Block, brain_index, init_para):
    ensembles = 1
    hp_num = len(para_ind)
    para = np.ones([ensembles, len(brain_index), hp_num]) * (init_para[para_ind-10].reshape(-1))
    pip = np.zeros([len(brain_index)*ensembles*hp_num, 2])
    pip[:, 0] = np.repeat(np.arange(len(brain_index)*ensembles), hp_num)
    pip[:, 1] = np.tile(para_ind, len(brain_index)*ensembles)
    Block.update_property(pip.astype(np.uint32), shift_rate * para.reshape(ensembles * len(brain_index) * hp_num).astype(np.float32))


def simulation(path, epsilon, ip):
    t1 = time.time()
    ensembles = 1
    # k = 10004232
    # epsilon = 200
    hp_range_rate = np.array([3, 3])
    para_ind = np.array([10, 12], dtype=int)
    # ip = '192.168.2.91:50051'
    block_path = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/v100"

    T = 450
    brain_num = 92
    noise_rate = 0.005
    steps = 800
    v_th = -50
    sample_number = 200000
    label = "assimilation"
    shift_rate = 1 + 0.1
    i_input = 1
    other_index = np.array([11, 13], dtype=int)

    Block = block_gpu(ip, "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/v100", noise_rate, 1., True)
    k = np.array(Block.total_neurons, dtype=np.int)
    print(k)
    arr = np.load(os.path.join(block_path, "block_0.npz"))
    property = arr["property"][:k]
    brain_index, nod_name, nod_sum = ge_brain_index(property)

    init_para = property[0, 10:14]
    # other_para(other_index, shift_rate, Block, brain_index, init_para)

    hp_real = property[0, 10:15].reshape([-1, 1])
    hp_real[-1] = i_input
    hp_range = np.concatenate((hp_real / hp_range_rate[0], hp_real * hp_range_rate[1]), axis=1)
    hp_num, hp, para, hp_low, hp_high, pip, hpip = ge_parameter(hp_range, para_ind, ensembles, brain_num, brain_index)

    w = np.load(os.path.join(path, "W.npy"))
    # path = path + 'nmda_gabab_shift'+str(shift_rate)
    path = path + 'noiserate'+str(noise_rate)
    # path = path + 'whole'
    os.makedirs(path, exist_ok=True)
    hhp = np.mean(w[:, :, :brain_num*hp_num], axis=1).reshape([T, brain_num, hp_num])
    hp = sigmoid_abs(hhp, hp_low, hp_high)  # * shift_rate
    bold_p = np.median(w[:, :, -5 * brain_num:], axis=1).reshape([-1, 5, brain_num])

    print(time.time()-t1)
    t1 = time.time()
    print(len(brain_index))
    sample_idx = load_if_exist(
        lambda: np.sort(np.random.choice(Block.total_neurons, sample_number, replace=False))
        if sample_number > 0 else np.arange(Block.total_neurons, dtype=np.uint32), path, "sample_idx")
    sample_number = sample_idx.shape[0]
    Block.update_property(pip, para.reshape(ensembles * k * hp_num).astype(np.float32))
    Block.set_samples(sample_idx)
    
    '''# fixed i input
    pipi = np.zeros([len(brain_index) * ensembles * 1, 2])
    pipi[:, 0] = np.repeat(np.arange(len(brain_index) * ensembles), 1)
    pipi[:, 1] = np.tile(np.array([2], dtype=int), len(brain_index) * ensembles)
    Block.update_property(pipi.astype(np.uint32), i_input * np.ones([ensembles * k * 1]).astype(np.float32))
    # fixed parameter
    #hp = hp[150:].mean(0)
    #Block.mul_property_by_subblk(hpip, hp.reshape(-1).astype(np.float32))
    '''

    FFreqs = np.zeros([T, steps, Block.total_subblks], dtype=np.uint32)
    bold1 = BOLD(epsilon=epsilon, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
    bold2 = BOLD(epsilon=epsilon, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
    bolds_out = np.zeros([T, 2, Block.total_subblks], dtype=np.float32)

    for ii in range((T - 1) // 100 + 1):
        nj = min(T - ii * 100, 100)
        Spike = np.zeros([nj, steps, sample_number], dtype=np.uint8)
        Vi = np.zeros([nj, steps, sample_number], dtype=np.float32)
        for j in range(nj):
            t12 = time.time()
            i = ii*100 + j
            Block.mul_property_by_subblk(hpip, hp[i, :].reshape(-1).astype(np.float32))
            t13 = time.time()
            Freqs, spike, vi = Block.run(steps, freqs=True, vmean=False, sample_for_show=True)
            spike &= (np.abs(vi - v_th)/np.abs(v_th) < 1e-5)
            for f in Freqs:
                bold_out1 = bold1.run(
                    np.maximum(f.astype(np.float32) / Block.neurons_per_subblk.astype(np.float32), 1e-5))
                bold_out2 = bold2.run(
                    np.maximum(f.astype(np.float32) / Block.neurons_per_subblk.astype(np.float32), 1e-5))
            bolds_out[i, 0] = bold_out1[-1, :]
            bolds_out[i, 1] = bold_out2[-1, :]
            bold1.s[:] = bold_p[i, 0]
            bold1.q[:] = bold_p[i, 1]
            bold1.v[:] = bold_p[i, 2]
            bold1.f_in[:] = bold_p[i, 3]
            t14 = time.time()
            FFreqs[i] = Freqs
            Spike[j] = spike
            Vi[j] = vi
            print(i, time.time() - t12 - (t14 - t13), t14 - t13, np.median(Freqs.astype(np.float32)/Block.neurons_per_subblk.astype(np.float32)*1000))
        np.save(os.path.join(path, "spike_{}_{}.npy".format(label, ii)), Spike)
        # np.save(os.path.join(path, "vi_{}_{}.npy".format(label, ii)), Vi)
    np.save(os.path.join(path, "Freqs_{}.npy".format(label)), FFreqs)
    np.save(os.path.join(path, "Bold_{}.npy".format(label)), bolds_out)
    print(time.time()-t1)
    print(FFreqs.sum(2).mean()/k)
    Block.shutdown()
    simulation_draw(w, bolds_out, T, path, brain_num)


def ge_parameter_exti(hp_range, para_ind, ensembles, brain_num, brain_index):
    # hp_range = np.array([[1,2],[3,4],[5,6],[7,8]])
    # para_ind = np.array([10,11,12,13], dtype=int)
    hp_num = len(para_ind)
    hp_low = np.tile(hp_range[para_ind - 10, 0], (brain_num, 1))  # shape = brain_num*hp_num
    hp_high = np.tile(hp_range[para_ind - 10, 1], (brain_num, 1))
    hp = np.linspace(hp_low, hp_high, 3*ensembles)[ensembles:-1*ensembles]  # shape = ensembles, brain_num, hp_num
    for i in range(hp_num):
        idx = np.random.choice(ensembles, ensembles, replace=False)
        hp[:, :, i] = hp[idx, :, i]
    para = np.random.exponential(np.ones([ensembles, len(brain_index), hp_num]))

    # p_exti_ind = np.nonzero((brain_index == 42) | (brain_index == 43))[0]
    p_exti_ind = np.nonzero((brain_index == 42) | (brain_index == 43) | (brain_index == 90) | (brain_index == 91))[0]
    print(p_exti_ind.shape)
    para_mask = np.zeros([ensembles, len(brain_index)])
    para_mask[:, p_exti_ind] = 1
    para[:, :, -1] = para[:, :, -1] * para_mask
    para_ind[-1] = 2

    pip = np.zeros([len(brain_index)*ensembles*hp_num, 2])
    pip[:, 0] = np.repeat(np.arange(len(brain_index)*ensembles), hp_num)
    pip[:, 1] = np.tile(para_ind, len(brain_index)*ensembles)
    hpip = np.zeros([brain_num*ensembles*hp_num, 2])
    hpip[:, 0] = np.repeat(np.arange(brain_num*ensembles), hp_num)
    hpip[:, 1] = np.tile(para_ind, brain_num*ensembles)
    return hp_num, log_abs(hp, hp_low, hp_high), para, hp_low, hp_high, pip.astype(np.uint32), hpip.astype(np.uint32)


def ensemble_system_exti(Block, Bold, steps, w, hp_fore, para, hp_num, brain_n, ensembles, k, hp_sigma, hp_low, hp_high, brain_index, nod_sum, bold_sigma, pip, hpip, hp_out):
    hp_transf_enkf = w[:, :brain_n*hp_num].reshape(ensembles, brain_n, hp_num) + np.sqrt(hp_sigma) * np.random.randn(ensembles, brain_n, hp_num)
    hp_transf_enkf[:, :, :(hp_num-1)] = hp_out.reshape(1, brain_n, hp_num-1)
    hp_enkf = sigmoid_abs(hp_transf_enkf, hp_low, hp_high)
    hp_delta = hp_enkf/sigmoid_abs(hp_fore.reshape(ensembles, brain_n, hp_num), hp_low, hp_high)
    # Block.mul_property_by_subblk(hpip, hp_delta.reshape(-1).astype(np.float32), accumulate=True)
    # Block.update_property(pip, (para*hp_enkf[:, brain_index, :]).reshape(ensembles * k * hp_num).astype(np.float32))
    Block.mul_property_by_subblk(hpip, hp_enkf.reshape(-1).astype(np.float32))

    act = Block.run(steps, freqs=True, vmean=False, sample_for_show=False)
    act = act/np.tile(nod_sum, ensembles)
    print(np.array(act).max(), np.array(act).min(), np.array(act).mean())

    df = Block.last_time_stat()
    for t in range(steps):
        bold = Bold.run(np.maximum(act[t], 1e-05))
    bold = bold + np.sqrt(bold_sigma) * np.random.randn(5, ensembles*brain_n)
    w_hat = np.concatenate((hp_transf_enkf.reshape(ensembles, brain_n*hp_num), (act[-1]).reshape([ensembles, brain_n]),
                            bold[0].reshape([ensembles, brain_n]), bold[1].reshape([ensembles, brain_n]),
                            bold[2].reshape([ensembles, brain_n]), bold[3].reshape([ensembles, brain_n]),
                            bold[4].reshape([ensembles, brain_n])), axis=1)
    return w_hat, hp_transf_enkf, df, act


def exti_da_show(W, data, da_old, T, path, brain_num):
    iteration = [i for i in range(T)]
    for i in range(brain_num):
        print("show"+str(i))
        fig = plt.figure(figsize=(8, 4), dpi=500)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(iteration, da_old[:T, i], 'r-', label='resting bold')
        if i == 42 or i == 43 or i == 90 or i == 91:
            ax1.plot(iteration, data[:T, i], 'k-', label='task bold')
        ax1.plot(iteration, np.mean(W[:T, :, -brain_num+i], axis=1), 'b-', label='after DA')
        plt.fill_between(iteration, np.mean(W[:T, :, -brain_num+i], axis=1) -
                         np.sqrt(np.var(W[:T, :, -brain_num+i], axis=1)), np.mean(W[:T, :, -brain_num+i], axis=1)
                         + np.sqrt(np.var(W[:T, :, -brain_num+i], axis=1)), color='b', alpha=0.2)
        plt.ylim((0.0, 0.1))
        plt.legend()
        plt.savefig(os.path.join(path, "boldv2"+str(i)+".png"))
        plt.close(fig)


def ext_i():
    start = time.time()
    ensembles = 2*37+2*0
    k = 10004232
    hp_sigma = 1
    bold_sigma = 1e-8

    degree = 100
    para_ind = np.array([10, 14], dtype=int)
    epsilon = 200
    hp_range_rate = np.array([3, 3])
    distributed_rate = 0.5

    block_path = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/block_all_v100"
    path = "./single10m/exti1_v1_degree"+str(degree)+str(para_ind)+str(epsilon)+str(hp_range_rate)+str(distributed_rate)+str(ensembles)
    os.makedirs(path, exist_ok=True)
    T = 400
    brain_num = 90
    noise_rate = 0.01
    steps = 800

    bold_y = loadmat("./AAL_ts_design.mat")
    bold_y = bold_y['run1_timecourse_band']
    bold_y = np.array(bold_y)[20:, :90]
    bold_y = 0.03 + 0.05 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())

    arr = np.load(os.path.join(block_path, "block_0.npz"))
    property = arr["property"][:k]

    brain_index, nod_name, nod_sum = ge_brain_index(property)
    hp_real = property[0, 10:15].reshape([-1, 1])
    hp_real[-1] = 1
    hp_range = np.concatenate((hp_real / hp_range_rate[0], hp_real * hp_range_rate[1]), axis=1)

    hp_num, hp, para, hp_low, hp_high, pip, hpip = ge_parameter_exti(hp_range, para_ind, ensembles, brain_num, brain_index)
    hp_out = np.load('./single10m/degree100[10]200[33]0.5/W.npy')[:, :, :brain_num].mean(1)
    hp_out[:, 42:44] = hp_out[100:, 42:44].mean(0)
    da_old = np.load('./single10m/degree100[10]200[33]0.5/W.npy')[:, :, -brain_num:].mean(1)
    hp[:, :, 0] = hp_out[0]
    print("ge_parameter:"+str(time.time()-start))

    Block = block_gpu('192.168.2.91:50051', block_path, noise_rate, 1., True)
    print("ge_Block:"+str(time.time()-start))
    
    Block.update_property(pip.reshape(-1, 4)[:, :2], para.reshape(-1, 2)[:, 0].astype(np.float32))
    Block.update_property(pip.reshape(-1, 4)[:, 2:], para.reshape(-1, 2)[:, 1].astype(np.float32))
    # Block.update_property(pip, para.reshape(ensembles * k * hp_num).astype(np.float32))
    Block.mul_property_by_subblk(hpip, sigmoid_abs(hp, hp_low, hp_high).reshape(-1).astype(np.float32))
    print("update_Block:"+str(time.time()-start))

    Bold = BOLD(epsilon=epsilon, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
    act = Block.run(steps, freqs=True, vmean=False, sample_for_show=False)

    print("run block:"+str(time.time()-start))
    for t in range(steps):
        bold = Bold.run(np.maximum(act[t]/np.tile(nod_sum, ensembles), 1e-05))
    print("run bold:" + str(time.time() - start))
    w = np.concatenate((hp.reshape([ensembles, brain_num*hp_num]),
                        (act[-1]/np.tile(nod_sum, ensembles)).reshape([ensembles, brain_num]),
                        bold[0].reshape([ensembles, brain_num]), bold[1].reshape([ensembles, brain_num]),
                        bold[2].reshape([ensembles, brain_num]), bold[3].reshape([ensembles, brain_num]),
                        bold[4].reshape([ensembles, brain_num])), axis=1)
    w_save = [w[:, :brain_num * (hp_num + 6)]]
    hp_fore = hp.copy().reshape([ensembles, brain_num * hp_num])
    print("begin da:" + str(time.time() - start))

    for t in range(T - 1):
        start = time.time()
        bold_y_t = bold_y[t].reshape([1, brain_num])
        w_hat, hp_fore, df, act = ensemble_system_exti(Block, Bold, steps, w, hp_fore, para, hp_num, brain_num,
                                                       ensembles, k, hp_sigma, hp_low, hp_high, brain_index, nod_sum,
                                                       bold_sigma, pip, hpip, hp_out[t+1])
        print("run da_es:" + str(time.time() - start))
        w_save.append(w_hat[:, :brain_num * (hp_num + 6)].copy())
        if t <= 8 or t % 10 == 8 or t == (T - 2):
            np.save(os.path.join(path, "W" + ".npy"), w_save)
            # df.to_csv(os.path.join(path, "data" + str(t) + ".csv"))
            # np.save(os.path.join(path, "act" + str(t) + ".npy"), act)
        w = distributed_kalman(w_hat, brain_num, ensembles, bold_sigma, bold_y_t, distributed_rate, hp_num)
        print("run da_dk:" + str(time.time() - start))
        Bold.s = torch.from_numpy(
            w[:, (hp_num + 1) * brain_num: (hp_num + 2) * brain_num].copy().reshape([ensembles * brain_num])).cuda()
        Bold.q = torch.from_numpy(np.maximum(w[:, (hp_num + 2) * brain_num: (hp_num + 3) * brain_num], 1e-05).reshape(
            [ensembles * brain_num])).cuda()
        Bold.v = torch.from_numpy(np.maximum(w[:, (hp_num + 3) * brain_num: (hp_num + 4) * brain_num], 1e-05).reshape(
            [ensembles * brain_num])).cuda()
        Bold.log_f_in = torch.from_numpy(
            np.maximum(w[:, (hp_num + 4) * brain_num: (hp_num + 5) * brain_num], -15).reshape(
                [ensembles * brain_num])).cuda()
        print("------------run da" + str(t) + ":" + str(time.time() - start))
    # da_show(np.array(w_save), bold_y, T, path, brain_num)
    show_hp(np.array(w_save), hp_low, hp_high, T, path, brain_num, para_ind)
    exti_da_show(np.array(w_save), bold_y, da_old, T, path, brain_num)
    Block.shutdown()


def read_spike(path):
    a = np.zeros([4, 100, 800, 200000], dtype=np.int8)
    for i in range(4):
        a[i] = np.load(os.path.join(path, "spike_assimilation_"+str(i)+".npy"))
    aa = a.reshape([400*800, 200000]).mean(0)*1000
    print(aa.max())
    return np.minimum(aa, 199.99)


def draw1():
    path = "./single10m"
    sample_number = 3
    acs = np.zeros([sample_number, 11])
    act_freq = np.zeros([sample_number, 200000])
    act_freq[0] = read_spike(path + "/degree100[1012]200[33]0.5")
    act_freq[1] = read_spike(path + "/exti3_v1_degree100[1014]200[33]0.558v1_solo")
    act_freq[2] = read_spike(path + "/exti3degree100[1014]100[33]0.548assimilation")

    print(act_freq.mean(1))
    for i in range(sample_number):
        act_number, act_count = np.unique(act_freq[i], return_counts=True)
        acs[i, 0] = act_count[0]
        print(acs[i, 0])
        for ii in range(len(act_number)):
            j = np.array(act_number[ii] // 20, dtype=np.int)
            acs[i, j + 1] += act_count[ii]
        acs[i, 1] = acs[i, 1] - acs[i, 0]
    print(acs.shape)
    acs = acs+1
    fig = mp.figure(figsize=(8, 4), dpi=500)
    ax1 = fig.add_subplot(1, 1, 1)

    x_names = ['0', '1-20', '21-40', '41-60', '61-80', '81-100', '101-120', '121-140', '141-160', '161-180', '181-200']
    ax1.semilogy(acs[0] / 200000, 'ro-', label='whole brain DA')
    ax1.semilogy(acs[1] / 200000, 'b.-', label='new method')
    ax1.semilogy(acs[2] / 200000, 'g.-', label='old method')
    ax1.set(xlabel='Neuronal Firing Rate(Hz)', ylabel='Proportion', title='Proportion of Neuronal Firing Rate')
    ax1.set_xticks(np.arange(len(x_names)))
    ax1.set_xticklabels(x_names)
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax1.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    ax1.yaxis.set_minor_locator(locmin)
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.tick_params(which='minor', width=1.0, labelsize=10)
    ax1.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')
    mp.setp(ax1.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    ax1.grid()
    ax1.legend()
    path = path + '/simulation0614'
    ax1.yaxis.grid(True, which='minor')
    mp.savefig(os.path.join(path, "v1solo.png"))
    mp.close(fig)


def lgn_ext_i():
    start = time.time()
    ensembles = 2 * 29 + 2 * 0
    k = 10001904
    hp_sigma = 1
    bold_sigma = 1e-6

    degree = 100
    para_ind = np.array([10, 12, 14], dtype=int)
    epsilon = 200
    hp_range_rate = np.array([3, 3])
    distributed_rate = 0.5

    block_path = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/block_v1lgn_solo90"
    path = "./single10m/n90lgn_exti3_v1_solo_degree" + str(degree) + str(para_ind) + str(epsilon) + str(
        hp_range_rate) + str(distributed_rate) + str(ensembles)
    os.makedirs(path, exist_ok=True)
    T = 450
    brain_num = 92
    noise_rate = 0.01
    steps = 800

    bold_y = loadmat("./single10m/AAL_task_LGN.mat")
    bold_y = bold_y['run1_timecourse_band']
    bold_y = np.array(bold_y)[:, :brain_num]
    bold_y = 0.03 + 0.05 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())

    arr = np.load("/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/v100/block_0.npz")
    property = arr["property"][:k]

    brain_index, nod_name, nod_sum = ge_brain_index(property)
    hp_real = property[0, 10:15].reshape([-1, 1])
    hp_real[-1] = 3
    hp_range = np.concatenate((hp_real / hp_range_rate[0], hp_real * hp_range_rate[1]), axis=1)

    hp_num, hp, para, hp_low, hp_high, pip, hpip = ge_parameter_exti(hp_range, para_ind, ensembles, brain_num,
                                                                     brain_index)
    hp_out = np.load('./single10m/lgn_degree100[1012]200[33]0.574/W.npy')[:, :, :(hp_num - 1)*brain_num].mean(1)
    # hp_out[:, 42:44] = hp_out[100:, 42:44].mean(0)
    # hp_out[:, 132:134] = hp_out[100:, 132:134].mean(0)
    hp_out[:, 84:88] = hp_out[100:, 84:88].mean(0)
    hp_out[:, 180:184] = hp_out[100:, 180:184].mean(0)
    da_old = np.load('./single10m/lgn_degree100[1012]200[33]0.574/W.npy')[:, :, -brain_num:].mean(1)
    hp[:, :, :(hp_num-1)] = hp_out[0].reshape(1, brain_num, hp_num-1)
    print("ge_parameter:" + str(time.time() - start))

    Block = block_gpu('192.168.2.91:50051', block_path, noise_rate, 1., True)
    print("ge_Block:" + str(time.time() - start))

    # Block.update_property(pip.reshape(-1, 4)[:, :2], para.reshape(-1, 2)[:, 0].astype(np.float32))
    # Block.update_property(pip.reshape(-1, 4)[:, 2:], para.reshape(-1, 2)[:, 1].astype(np.float32))
    Block.update_property(pip, para.reshape(-1).astype(np.float32))
    Block.mul_property_by_subblk(hpip, sigmoid_abs(hp, hp_low, hp_high).reshape(-1).astype(np.float32))
    print("update_Block:" + str(time.time() - start))

    Bold = BOLD(epsilon=epsilon, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
    act = Block.run(steps, freqs=True, vmean=False, sample_for_show=False)

    print("run block:" + str(time.time() - start))
    for t in range(steps):
        bold = Bold.run(np.maximum(act[t] / np.tile(nod_sum, ensembles), 1e-05))
    print("run bold:" + str(time.time() - start))
    w = np.concatenate((hp.reshape([ensembles, brain_num * hp_num]),
                        (act[-1] / np.tile(nod_sum, ensembles)).reshape([ensembles, brain_num]),
                        bold[0].reshape([ensembles, brain_num]), bold[1].reshape([ensembles, brain_num]),
                        bold[2].reshape([ensembles, brain_num]), bold[3].reshape([ensembles, brain_num]),
                        bold[4].reshape([ensembles, brain_num])), axis=1)
    w_save = [w[:, :brain_num * (hp_num + 6)]]
    hp_fore = hp.copy().reshape([ensembles, brain_num * hp_num])
    print("begin da:" + str(time.time() - start))

    for t in range(T - 1):
        start = time.time()
        bold_y_t = bold_y[t].reshape([1, brain_num])
        w_hat, hp_fore, df, act = ensemble_system_exti(Block, Bold, steps, w, hp_fore, para, hp_num, brain_num,
                                                       ensembles, k, hp_sigma, hp_low, hp_high, brain_index, nod_sum,
                                                       bold_sigma, pip, hpip, hp_out[t + 1])
        print("run da_es:" + str(time.time() - start))
        print(hp_fore.max(), hp_fore.min())
        w_save.append(w_hat[:, :brain_num * (hp_num + 6)].copy())
        if t <= 8 or t % 30 == 28 or t == (T - 2):
            np.save(os.path.join(path, "W" + ".npy"), w_save)
            # df.to_csv(os.path.join(path, "data" + str(t) + ".csv"))
            # np.save(os.path.join(path, "act" + str(t) + ".npy"), act)
        w = distributed_kalman(w_hat, brain_num, ensembles, bold_sigma, bold_y_t, distributed_rate, hp_num)
        print("run da_dk:" + str(time.time() - start))
        '''
        Bold.s = torch.from_numpy(
            w[:, (hp_num + 1) * brain_num: (hp_num + 2) * brain_num].copy().reshape([ensembles * brain_num])).cuda()
        Bold.q = torch.from_numpy(np.maximum(w[:, (hp_num + 2) * brain_num: (hp_num + 3) * brain_num], 1e-05).reshape(
            [ensembles * brain_num])).cuda()
        Bold.v = torch.from_numpy(np.maximum(w[:, (hp_num + 3) * brain_num: (hp_num + 4) * brain_num], 1e-05).reshape(
            [ensembles * brain_num])).cuda()
        Bold.log_f_in = torch.from_numpy(
            np.maximum(w[:, (hp_num + 4) * brain_num: (hp_num + 5) * brain_num], -15).reshape(
                [ensembles * brain_num])).cuda()
        '''
        Bold.s = w[:, (hp_num + 1) * brain_num: (hp_num + 2) * brain_num].copy().reshape([ensembles * brain_num])
        Bold.q = np.maximum(w[:, (hp_num + 2) * brain_num: (hp_num + 3) * brain_num], 1e-05).reshape(
            [ensembles * brain_num])
        Bold.v = np.maximum(w[:, (hp_num + 3) * brain_num: (hp_num + 4) * brain_num], 1e-05).reshape(
            [ensembles * brain_num])
        Bold.f_in = np.maximum(w[:, (hp_num + 4) * brain_num: (hp_num + 5) * brain_num], -15).reshape(
                [ensembles * brain_num])
        print("------------run da" + str(t) + ":" + str(time.time() - start))
    # da_show(np.array(w_save), bold_y, T, path, brain_num)
    show_hp(np.array(w_save), hp_low, hp_high, T, path, brain_num, para_ind)
    exti_da_show(np.array(w_save), bold_y, da_old, T, path, brain_num)
    Block.shutdown()


def block_double():
    path_in = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/v100"
    # path_out = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/v100_block_divide"
    file = np.load(os.path.join(path_in, 'block_0.npz'))

    property = file["property"]
    output_neuron_idx = file["output_neuron_idx"]
    input_block_idx = file["input_block_idx"]
    input_neuron_idx = file["input_neuron_idx"]
    input_channel_offset = file["input_channel_offset"]
    weight = file["weight"]

    property[:, 3] += 92
    output_neuron_idx += 3333968
    input_neuron_idx += 10001904

    property = np.concatenate((file["property"], property), axis=0)
    output_neuron_idx = np.concatenate((file["output_neuron_idx"], output_neuron_idx), axis=0)
    input_block_idx = np.concatenate((file["input_block_idx"], input_block_idx), axis=0)
    input_neuron_idx = np.concatenate((file["input_neuron_idx"], input_neuron_idx), axis=0)
    input_channel_offset = np.concatenate((file["input_channel_offset"], input_channel_offset), axis=0)
    weight = np.concatenate((file["weight"], weight), axis=0)

    output_neuron_idx = np.ascontiguousarray(output_neuron_idx.astype(np.uint32))
    input_block_idx = np.ascontiguousarray(input_block_idx.astype(np.int16))
    input_neuron_idx = np.ascontiguousarray(input_neuron_idx.astype(np.uint32))
    input_channel_offset = np.ascontiguousarray(input_channel_offset.astype(np.uint8))
    property = np.ascontiguousarray(property)
    weight = np.ascontiguousarray(weight)

    np.savez(os.path.join(path_in, 'block_0double.npz'), property=property,
             output_neuron_idx=output_neuron_idx,
             input_block_idx=input_block_idx,
             input_neuron_idx=input_neuron_idx,
             input_channel_offset=input_channel_offset,
             weight=weight)


def lgn_simulation(path, epsilon, ip, falpha=None, fbeta=None):
    t1 = time.time()
    ensembles = 1
    # k = 10004232
    # epsilon = 200
    hp_range_rate = np.array([3, 3])
    para_ind = np.array([10, 12, 14], dtype=int)
    # ip = '192.168.2.91:50051'
    # block_path = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/block_split90"
    # block_path = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/v10090"
    block_path = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/block_" + str(falpha) + str(fbeta)

    T = 450
    brain_num = 92
    noise_rate = 0.01
    steps = 800
    v_th = -50
    sample_number = 200000
    label = "assimilation"
    shift_rate = 1 + 0.1
    i_input = 0.2
    other_index = np.array([11, 12, 13], dtype=int)

    Block = block_gpu(ip, block_path, noise_rate, 1., True)
    k = np.array(Block.total_neurons, dtype=np.int)
    print(k)
    arr = np.load("/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/lgn/v100/block_0.npz")
    property = arr["property"][:k]
    brain_index, nod_name, nod_sum = ge_brain_index(property)

    init_para = property[0, 10:14]
    # other_para(other_index, shift_rate, Block, brain_index, init_para)

    hp_real = property[0, 10:15].reshape([-1, 1])
    hp_real[-1] = 1
    hp_range = np.concatenate((hp_real / hp_range_rate[0], hp_real * hp_range_rate[1]), axis=1)
    hp_num, hp, para, hp_low, hp_high, pip, hpip = ge_parameter_exti(hp_range, para_ind, ensembles, brain_num, brain_index)

    w = np.load(os.path.join(path, "W.npy"))
    # path = path + 'ampa_i_shift'+str(i_input)
    path = path + str(falpha) + str(fbeta)
    os.makedirs(path, exist_ok=True)
    hhp = np.mean(w[:, :, :brain_num * hp_num], axis=1).reshape([T, brain_num, hp_num])
    hp = sigmoid_abs(hhp, hp_low, hp_high)
    # hp = np.tile(sigmoid_abs(hhp, hp_low, hp_high).mean(0), T).reshape([T, brain_num, hp_num])
    hp[:, np.array([42, 43, 90, 91], dtype=int)] = sigmoid_abs(hhp, hp_low, hp_high)[:, np.array([42, 43, 90, 91], dtype=int)]  # * shift_rate
    bold_p = np.median(w[:, :, -5 * brain_num:], axis=1).reshape([-1, 5, brain_num])

    print(time.time() - t1)
    t1 = time.time()
    print(len(brain_index))
    sample_idx = load_if_exist(
        lambda: np.sort(np.random.choice(Block.total_neurons, sample_number, replace=False))
        if sample_number > 0 else np.arange(Block.total_neurons, dtype=np.uint32), path, "sample_idx")
    sample_number = sample_idx.shape[0]
    Block.update_property(pip, para.reshape(ensembles * k * hp_num).astype(np.float32))
    Block.set_samples(sample_idx)

    '''# fixed i input
    pipi = np.zeros([len(brain_index) * ensembles * 1, 2])
    pipi[:, 0] = np.repeat(np.arange(len(brain_index) * ensembles), 1)
    pipi[:, 1] = np.tile(np.array([2], dtype=int), len(brain_index) * ensembles)
    Block.update_property(pipi.astype(np.uint32), i_input * np.ones([ensembles * k * 1]).astype(np.float32))
    # fixed parameter
    #hp = hp[150:].mean(0)
    #Block.mul_property_by_subblk(hpip, hp.reshape(-1).astype(np.float32))
    '''

    FFreqs = np.zeros([T, steps, Block.total_subblks], dtype=np.uint32)
    bold1 = BOLD(epsilon=epsilon, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
    bold2 = BOLD(epsilon=epsilon, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
    bolds_out = np.zeros([T, 2, Block.total_subblks], dtype=np.float32)

    for ii in range((T - 1) // 100 + 1):
        nj = min(T - ii * 100, 100)
        Spike = np.zeros([nj, steps, sample_number], dtype=np.uint8)
        Vi = np.zeros([nj, steps, sample_number], dtype=np.float32)
        for j in range(nj):
            t12 = time.time()
            i = ii * 100 + j
            Block.mul_property_by_subblk(hpip, hp[i, :].reshape(-1).astype(np.float32))
            t13 = time.time()
            Freqs, spike, vi = Block.run(steps, freqs=True, vmean=False, sample_for_show=True)
            spike &= (np.abs(vi - v_th) / np.abs(v_th) < 1e-5)
            for f in Freqs:
                bold_out1 = bold1.run(
                    np.maximum(f.astype(np.float32) / Block.neurons_per_subblk.astype(np.float32), 1e-5))
                bold_out2 = bold2.run(
                    np.maximum(f.astype(np.float32) / Block.neurons_per_subblk.astype(np.float32), 1e-5))
            bolds_out[i, 0] = bold_out1[-1, :]
            bolds_out[i, 1] = bold_out2[-1, :]
            bold1.s[:] = bold_p[i, 0]
            bold1.q[:] = bold_p[i, 1]
            bold1.v[:] = bold_p[i, 2]
            bold1.f_in[:] = bold_p[i, 3]
            t14 = time.time()
            FFreqs[i] = Freqs
            Spike[j] = spike
            Vi[j] = vi
            print(i, time.time() - t12 - (t14 - t13), t14 - t13,
                  np.median(Freqs.astype(np.float32) / Block.neurons_per_subblk.astype(np.float32) * 1000))
        # np.save(os.path.join(path, "spike_{}_{}.npy".format(label, ii)), Spike)
        # np.save(os.path.join(path, "vi_{}_{}.npy".format(label, ii)), Vi)
    np.save(os.path.join(path, "Freqs_{}.npy".format(label)), FFreqs)
    np.save(os.path.join(path, "Bold_{}.npy".format(label)), bolds_out)
    print(time.time() - t1)
    print(FFreqs.sum(2).mean() / k)
    Block.shutdown()
    simulation_draw(w, bolds_out, T, path, brain_num)


if __name__ == "__main__":
    main()
    # lgn_ext_i()
    # transfer_to_relative('./single10m/block_all', 30, './single10m/block_all_n')
    #block_divide(0, 1.5)
    # block_divide(0.5, 0)
    # block_divide(0.5, 0.5)
    #block_divide(0.5, 1)
    #block_divide(0.5, 1.5)
    #block_divide(0.5, 2)
    # block_v1_divide()
    # simulation("./single10m/lgn_degree100[1012]200[33]0.574", 200, '192.168.2.91:50051')
    # draw1()
    # block_double()
    # lgn_simulation("./single10m/nlgn_exti3_v1_solo_degree100[101214]200[33]0.574", 200, '192.168.2.94:50051', 2, 0.5)
    '''
    w = np.load('./single10m/degree100[1014]200[33]0.5/W.npy')
    bolds_out = np.load('./single10m/exti3_v1solo_degree100[1014]200[33]0.548lienao/Bold_assimilation.npy')
    simulation_draw(w, bolds_out, 400, './single10m/exti3_v1solo_degree100[1014]200[33]0.548lienao', 90)
    ''''''
    bolds_out1 = np.load('./single10m/exti3_v1solo_degree100[1014]200[33]0.548whole/Bold_assimilation.npy')
    bolds_out2 = np.load('./single10m/exti3_v1solo_degree100[1014]200[33]0.548lienao/Bold_assimilation.npy')
    simulation_draw(bolds_out1, bolds_out2, 400, './single10m/exti3_v1solo_degree100[1014]200[33]0.548lienao', 90)
    '''
