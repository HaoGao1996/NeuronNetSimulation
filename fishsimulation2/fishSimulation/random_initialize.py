import torch
import numpy as np
import os
from multiprocessing.pool import Pool as pool
from multiprocessing.pool import ThreadPool as Thpool
import re
from numba import jit #加速器

# random initialize the connections

# Each block should be a sparse tensor with the shape of [K, N, K, 4] and storage as name "block_n.pkl",
# where K is the block size, N is the total block number, 4 stands for 4 type of connections(AMPA, NMPA, GABAa, GABAb),
# and the slice [k, :, :, :] of the tensor stands for the weight of the incoming connections of the neuro k in block n.

# Also the following limitations are guaranteed:
# 1. type of connections order is [AMPA, NMPA, GABAa, GABAb].
# 2. each neuro has a type (E or I) and:
#    E type neuro has and only has the AMPA and NMPA out-coming connections for each consumer neuro.
#    I type neuro has and only has the GABAa and GABAb out-coming connections for each consumer neuro.
# 3. The order of the dim K must be [E neurons, I neurons]
# 4. for "block_n.pkl", for any k, the slice [k, n, k, :] of the tensor should be zeros(no self connection).

def load_if_exist(only_load, func, *args):
    path = os.path.join(*args)
    if only_load:
        while True:
            if os.path.exists(path + ".npy"):
                try:
                    return np.load(path + ".npy")
                except:
                    continue
    else:
        if os.path.exists(path + ".npy"):
            out = np.load(path + ".npy")
        else:
            print('running generation')
            out = func()
            print('done!')
            np.save(path, out)
        return out


def transfer(path, bid, output_path=None):
    if output_path is None:
        output_path = path
    path = os.path.join(path, "block_{}.npz".format(bid))
    output_path = os.path.join(output_path, "block_{}".format(bid))

    file = np.load(path)
    property = file["property"]
    idx = file["idx"]
    weight = file["weight"]
    idx = idx[:, ::2]
    weight = weight.reshape([-1, 2])

    output_neuron_idx = np.ascontiguousarray(idx[0, :].astype(np.uint32))
    input_block_idx = np.ascontiguousarray(idx[1, :].astype(np.int16))
    input_block_idx -= bid
    input_neuron_idx = np.ascontiguousarray(idx[2, :].astype(np.uint32))
    input_channel_offset = np.ascontiguousarray(idx[3, :].astype(np.uint8))

    weight = np.ascontiguousarray(weight)
    np.savez(output_path,
             property=property,
             output_neuron_idx=output_neuron_idx,
             input_block_idx=input_block_idx,
             input_neuron_idx=input_neuron_idx,
             input_channel_offset=input_channel_offset,
             weight=weight)


def transfer_to_relative(path, bid, output_path=None):
    if output_path is None:
        output_path = path
    path = os.path.join(path, "block_{}.npz".format(bid))
    output_path = os.path.join(output_path, "block_{}".format(bid))

    file = np.load(path)
    file = {key:file[key] for key in file.keys()}
    file['input_block_idx'] = np.ascontiguousarray(file['input_block_idx'].astype(np.int16))
    file['input_block_idx'] -= bid
    np.savez(output_path, **file)


def generate_block_node_property(E_number=int(8e5),
                            I_number=int(2e5),
                            I_extern_Input = 0,
                            sub_block_idx=0,
                            C = 1,
                            T_ref = 5,
                            g_Li = 0.001,
                            V_L = -75,
                            V_th = -50,
                            V_reset = -65,
                            g_ui = (5/275, 5/4000, 3/30, 3/730),
                            V_ui = (0, 0, -70, -100),
                            tao_ui = (2, 40, 10, 50),
                            s = 0, e = -1):

    # each node contain such property:
    #          E/I, blocked_in_stat, I_extern_Input, sub_block_idx, C, T_ref, g_Li, V_L, V_th, V_reset, g_ui, V_ui, tao_ui
    #   size:  1,   1,               1,                1,           1, 1,     1,    1,   1,    1,       4     4,    4
    #   dtype: b,   b,               f,                i,           f, f,     f,    f,   f,    f,       f,    f,    f
    # b means bool(although storage as float), f means float.

    # this function support broadcast, e.g, C can be a scalar for a total block or a [E_number, I_number] tensor for total nodes.
    if e == -1:
        e = E_number + I_number
    assert 0 <= s < e <= E_number + I_number

    property = np.zeros([e - s, 22], dtype=np.float32)
    E_thresh = E_number - s if E_number > s else 0
    property[:E_thresh, 0] = 1
    property[E_thresh:, 0] = 0

    property[:, 1] = 0

    property[:, 2] = I_extern_Input

    property[:, 3] = sub_block_idx
    property[:, 4] = C
    property[:, 5] = T_ref
    property[:, 6] = g_Li
    property[:, 7] = V_L
    property[:, 8] = V_th
    property[:, 9] = V_reset

    g_ui = g_ui if isinstance(g_ui, np.ndarray) else np.array(g_ui)
    property[:, 10:14] = g_ui

    V_ui = V_ui if isinstance(V_ui, np.ndarray) else np.array(V_ui)
    property[:, 14:18] = V_ui

    tao_ui = tao_ui if isinstance(tao_ui, np.ndarray) else np.array(tao_ui)

    property[:, 18:22] = tao_ui

    return property

@jit(nogil=True, nopython=True)
def get_k_idx(max_k, num, except_idx):
    if except_idx < 0:
        assert num <= max_k
        if num == max_k:
            return np.arange(0, max_k)
    elif except_idx is not None:
        assert num < max_k
        if num == max_k - 1:
            return np.concatenate((np.arange(0, except_idx), np.arange(except_idx+1, num)))#输出除了except_idx之外的其他连续数列，从0到num

    while True:
        k_idx = np.unique(np.random.randint(0, max_k, num*2))#随机生成一个2*num规模的随机整数列，去掉重复值，取值为0-max_k
        k_idx = k_idx[np.random.permutation(k_idx.shape[0])]#对上述产生的序列进行随机打乱
        if except_idx is not None:
            k_idx = k_idx[k_idx != except_idx] #删掉except_idx
        k_idx = k_idx[:num]#截取num个元素
        if k_idx.shape[0] == num:
            break
    return k_idx


def connect_for_single_sparse_block(block_idx: object, k: object, extern_input_rate: object, extern_input_k_sizes: object, degree: object = int(1e3),
                                    init_min: object = 0,
                                    init_max: object = 1,
                                    s: object = 0,
                                    e: object = -1) -> object:
    # extern_input_rate only works for E neurons.

    # this function is pertty slow.

    # we assume that the permution

    if e == -1:
        e = k
    assert 0 <= s < e <= k

    print("length:", e - s, "degree:", degree)
    print("generating weight")
    connect_weight = np.random.rand(e - s, degree, 2).astype(np.float32) * (init_max - init_min) + init_min
    print("generating idx")

    E_neuron_thresh = extern_input_k_sizes[block_idx]

    # connect_weight[is_E_neuron == 0] *= 4

    output_neuron_idx = np.broadcast_to(np.arange(s, e, dtype=np.uint32)[:, None], (e-s, degree))

    _extern_input_k_sizes = np.array(extern_input_k_sizes, dtype=np.int64)
    extern_input_rate = np.add.accumulate(extern_input_rate)[:-1]

    @jit(nogil=True, nopython=True)
    def _run(i):
        input_block_idx = np.zeros(degree, dtype=np.int16)#初始化
        input_channel_offset = np.zeros(degree, dtype=np.uint8)#初始化
        while True:
            k_idx = get_k_idx(k, degree, i)#不能与自己连接
            k_idx_is_E = (k_idx < E_neuron_thresh)##与E_neuron产生随机的连接关系？
            k_idx_is_I = (k_idx >= E_neuron_thresh)##与I_neuron产生随机的连接关系？
            if k_idx_is_I.shape[0] > 0:
                break
        # connect_weight[i, k_idx_is_I] *= connect_weight[i, k_idx_is_E].sum() / connect_weight[i, k_idx_is_I].sum()

        input_block_idx[k_idx_is_I] = block_idx#设置I_neuron

        r = np.random.rand(np.count_nonzero(k_idx_is_E))
        E_in_comming = np.searchsorted(extern_input_rate, r, 'right').astype(np.int16)

        input_block_idx[k_idx_is_E] = E_in_comming#设置E_neuron

        input_neuron_idx = k_idx.astype(np.uint32)
        for _idx, max_idx in enumerate(_extern_input_k_sizes):
            if _idx != block_idx:
                extern_incomming_idx = (input_block_idx == _idx).nonzero()[0]
                extern_outcomming_idx = get_k_idx(max_idx, extern_incomming_idx.shape[0], -1)
                input_neuron_idx[extern_incomming_idx] = extern_outcomming_idx

        input_channel_offset[k_idx_is_E] = 0 #设置offset
        input_channel_offset[k_idx_is_I] = 2 #设置offset

        input_block_idx -= block_idx
        return input_block_idx, input_neuron_idx, input_channel_offset

    with Thpool() as p:#多线程
        input_block_idx, input_neuron_idx, input_channel_offset = tuple(zip(*p.map(_run, range(s, e))))
    input_block_idx = np.concatenate(input_block_idx)
    input_neuron_idx = np.concatenate(input_neuron_idx)
    input_channel_offset = np.concatenate(input_channel_offset)
    output_neuron_idx = output_neuron_idx.reshape([-1])
    connect_weight = connect_weight.reshape([-1, 2])

    print("done", e - s)
    return output_neuron_idx, input_block_idx, input_neuron_idx, input_channel_offset, connect_weight


def _process_dti(i):
    global block_connect_prob
    global block_node_init_kwards
    global extern_input_k_sizes
    global degree
    global init_min
    global init_max
    global dtype
    global perfix

    def check_if_exist():
        for d in dtype:
            print(d)
            new_perfix = perfix if len(dtype) == 0 else os.path.join(perfix, d)
            os.makedirs(new_perfix, exist_ok=True)
            if not os.path.exists(os.path.join(new_perfix, "block_{}.npz".format(i))):
                return False
        return True

    if check_if_exist():
        print("skip", i)
        return
    print("processing", i)
    prob = block_connect_prob[i, :]
    assert np.abs(1 - prob.sum()) < 1e-4#概率加起来基本上等于1
    if isinstance(block_node_init_kwards, list):#剥皮
        block_node_init_kward = block_node_init_kwards[i]
    else:
        block_node_init_kward = block_node_init_kwards
    assert isinstance(block_node_init_kward, dict)
    property = generate_block_node_property(sub_block_idx=i, **block_node_init_kward)
    output_neuron_idx, input_block_idx, input_neuron_idx, input_channel_offset, weight \
        = connect_for_single_sparse_block(i, block_node_init_kward['E_number'] + block_node_init_kward['I_number'],
                                             prob,
                                             extern_input_k_sizes=extern_input_k_sizes,
                                             degree=degree,
                                             init_min=init_min,
                                             init_max=init_max)

    if isinstance(dtype, str):
        dtype = [dtype]

    for d in dtype:
        new_perfix = perfix if len(dtype) == 0 else os.path.join(perfix, d)
        os.makedirs(new_perfix, exist_ok=True)

        if d == "single":
            _weight = weight
        elif d == 'half':
            _weight = weight.astype(np.float16)
        else:
            raise ValueError
        np.savez(os.path.join(new_perfix, "block_{}".format(i)),
                 property=np.ascontiguousarray(property),
                 output_neuron_idx=np.ascontiguousarray(output_neuron_idx),
                 input_block_idx=np.ascontiguousarray(input_block_idx),
                 input_neuron_idx=np.ascontiguousarray(input_neuron_idx),
                 input_channel_offset=np.ascontiguousarray(input_channel_offset),
                 weight=np.ascontiguousarray(_weight))

    print("done! ", i)


def _init(_block_connect_prob,
          _block_node_init_kwards,
          _extern_input_k_sizes,
          _degree,
          _init_min,
          _init_max,
          _dtype,
          _perfix):
    global block_connect_prob
    global block_node_init_kwards
    global extern_input_k_sizes
    global degree
    global init_min
    global init_max
    global dtype
    global perfix

    block_connect_prob = _block_connect_prob
    block_node_init_kwards = _block_node_init_kwards
    extern_input_k_sizes = _extern_input_k_sizes
    degree = _degree
    init_min = _init_min
    init_max = _init_max
    dtype = _dtype
    perfix = _perfix

    np.random.seed()


def connect_for_multi_sparse_block(block_connect_prob, block_node_init_kwards=None, E_number=None, I_number=None, degree=int(1e3), init_min=0, init_max=1, perfix=None, dtype="single"):
    print("connect_for_multi_sparse_block")
    assert isinstance(block_connect_prob, torch.Tensor) and \
           len(block_connect_prob.shape) == 2 and \
           block_connect_prob.shape[0] == block_connect_prob.shape[1]
    # block_connect_prob should be a [N, N] tensor
    N = block_connect_prob.shape[0]
    block_connect_prob = block_connect_prob.numpy()

    #参数设置整理
    block_node_init_kwards = {} if block_node_init_kwards is None else block_node_init_kwards
    if E_number is not None:
        block_node_init_kwards["E_number"] = E_number
    if I_number is not None:
        block_node_init_kwards["I_number"] = I_number

    if isinstance(block_node_init_kwards, dict):
        extern_input_k_sizes = [block_node_init_kwards["E_number"]] * N
    elif isinstance(block_node_init_kwards, list):
        extern_input_k_sizes = [b["E_number"] for b in block_node_init_kwards]
    else:
        raise ValueError

    print('total {} blocks'.format(N))
    if perfix is None:
        print("no prefix")
        def _out():
            if isinstance(block_node_init_kwards, dict):
                number = [block_node_init_kwards['E_number'] + block_node_init_kwards['I_number']] * N
            elif isinstance(block_node_init_kwards, list):
                number = [b['E_number'] + b['I_number'] for b in block_node_init_kwards]
            else:
                raise ValueError

            bases = np.add.accumulate(np.array(number, dtype=np.int64))
            bases = np.concatenate([np.array([0], dtype=np.int64), bases])

            def prop(i, s, e):
                block_node_init_kward = block_node_init_kwards[i] if isinstance(block_node_init_kwards, list) else block_node_init_kwards#剥壳
                return generate_block_node_property(sub_block_idx=i, s=s, e=e, **block_node_init_kward)

            def conn(i, s, e):
                prob = block_connect_prob[i, :]
                assert np.abs(1 - prob.sum()) < 1e-4
                step = int(1e6)
                for _s in range(s, e, step):
                    _e = min(_s+step, e)
                    output_neuron_idx, input_block_idx, input_neuron_idx, input_neuron_offset, connect_weight =\
                        connect_for_single_sparse_block(i, bases[i+1] - bases[i],
                                                          prob,
                                                          s=_s,
                                                          e=_e,
                                                          extern_input_k_sizes=extern_input_k_sizes,
                                                          degree=degree,
                                                          init_min=init_min,
                                                          init_max=init_max)

                    output_neuron_idx = output_neuron_idx.astype(np.int64)
                    input_neuron_idx = input_neuron_idx.astype(np.int64)

                    output_neuron_idx += bases[i].astype(output_neuron_idx.dtype)
                    input_neuron_idx += bases[i + input_block_idx].astype(input_neuron_idx.dtype)
                    yield output_neuron_idx, input_neuron_idx, input_neuron_offset, connect_weight
            return prop, conn, bases
        return _out
    else:
        if not os.path.exists(perfix):
            os.makedirs(perfix)
        with pool(8, initializer=_init, initargs=(block_connect_prob,
                                               block_node_init_kwards,
                                               extern_input_k_sizes,
                                               degree,
                                               init_min,
                                               init_max,
                                               dtype,
                                               perfix)) as p:
            p.map(_process_dti, range(0, N, 1), chunksize=1)
        print("dtype: {}".format(dtype))
        print("dtype—type: {}".format(type(dtype)))
        print('total done!')
        return


def connect_for_block(path, dense=True, bases=None):
    block_name = re.compile('block_[0-9]*.npz')
    block_length = len([name for name in os.listdir(path) if block_name.fullmatch(name)])#返回指定目录中npz文件的数量
    #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
    if bases is None:
        bases = [0]
        for i in range(block_length):
            pkl_path = os.path.join(path, "block_{}.npz".format(i))
            file = np.load(pkl_path)
            bases.append(bases[-1] + file["property"].shape[0])

        bases = np.array(bases, dtype=np.int64)

    if dense:#如果是一般的矩阵
        weights = []
        indices = []
        sizes = []
        properties = []
        for i in range(block_length):#开始读取每个block中的数据数据
            pkl_path = os.path.join(path, "block_{}.npz".format(i))
            file = np.load(pkl_path)
            properties.append(file["property"])
            output_neuron_idx, input_block_idx, input_neuron_idx, input_channel_offset, weight = \
                tuple(file[name] for name in ["output_neuron_idx", "input_block_idx", "input_neuron_idx", "input_channel_offset", "weight"])
            idx = np.stack([input_channel_offset.astype(np.uint32),
                            output_neuron_idx.astype(np.uint32),
                            (bases[(input_block_idx + i)] + input_neuron_idx).astype(np.uint32)])
            weight = weight.reshape([-1])
            idx = np.stack([idx, idx], axis=-1)
            idx[0, :, 1] = idx[0, :, 0] + 1
            idx = idx.reshape([3, -1])
            size = [4, np.max(idx[1])+1, np.max(idx[2])+1]

            indices.append(idx)
            weights.append(weight)
            sizes.append(size)
        size = tuple(np.max(np.array(sizes), axis=0)[1:].tolist())
        property = torch.cat([torch.from_numpy(property) for property in properties])
        weight = torch.cat([torch.sparse.FloatTensor(
                    torch.from_numpy(indices[i].astype(np.int64)),
                    torch.from_numpy(weights[i]),
                    torch.Size([4, bases[i+1] - bases[i], size[1]])) for i in range(block_length)], dim=2)
        assert property.shape[0] == weight.shape[1]
        assert weight.shape[2] == weight.shape[1]
        print(property.shape, weight.shape)
        return property, weight#返回参数
    else:
        def conn(i, s, e):
            pkl_path = os.path.join(path, "block_{}.npz".format(i))
            file = np.load(pkl_path)
            output_neuron_idx, input_block_idx, input_neuron_idx, input_channel_offset, weight = \
                tuple(file[name] for name in ["output_neuron_idx", "input_block_idx", "input_neuron_idx", "input_channel_offset", "weight"])
            selection_idx = np.logical_and(output_neuron_idx >= s, output_neuron_idx < e).nonzero()[0]

            output_neuron_idx = np.take(output_neuron_idx, selection_idx, axis=0)
            input_block_idx = np.take(input_block_idx, selection_idx, axis=0)
            input_channel_offset = np.take(input_channel_offset, selection_idx, axis=0)
            input_neuron_idx = np.take(input_neuron_idx, selection_idx, axis=0)
            weight = np.take(weight, selection_idx, axis=0)

            output_neuron_idx = output_neuron_idx.astype(np.int64)
            input_neuron_idx = input_neuron_idx.astype(np.int64)

            output_neuron_idx += bases[i].astype(output_neuron_idx.dtype)
            input_neuron_idx += bases[i + input_block_idx].astype(input_neuron_idx.dtype)

            yield output_neuron_idx, input_neuron_idx, input_channel_offset, weight

        def prop(i, s, e):
            pkl_path = os.path.join(path, "block_{}.npz".format(i))
            file = np.load(pkl_path)
            property = file["property"]
            assert 0<=s<e<=property.shape[0]
            return property[s:e].copy()

        return prop, conn, bases


def add_debug(debug_block_path, prop, conn, _dti_block_thresh, debug_idx_path, only_load):
    debug_prop, debug_conn, debug_block_thresh = connect_for_block(debug_block_path, dense=False)
    main_size = _dti_block_thresh[-1]

    def generate_debug_permutation_idx():
        debug_permutation_idx = (np.random.permutation(main_size + debug_block_thresh[-1])).astype(np.int64)
        debug_permutation_idx[debug_permutation_idx < main_size] = np.arange(main_size, dtype=np.int64)
        return debug_permutation_idx

    debug_permutation_idx = load_if_exist(only_load, generate_debug_permutation_idx, debug_idx_path, 'debug_permutation_idx')
    debug_recover_idx = np.argsort(debug_permutation_idx)

    dti_block_thresh = _dti_block_thresh.copy()
    dti_block_thresh[:-1] = debug_recover_idx[_dti_block_thresh[:-1]]
    dti_block_thresh[-1] = main_size + debug_block_thresh[-1]

    debug_selection_idx = debug_recover_idx[main_size:]

    def add_debug_prop(_prop):
        debug_property = np.concatenate([debug_prop(i, 0, e - s) for i, (s, e) in enumerate(zip(debug_block_thresh[:-1],
                                                                                            debug_block_thresh[1:]))])
        debug_property[:, 1] = 1

        def prop(i, s, e):
            select_start = dti_block_thresh[i] + s
            select_end = dti_block_thresh[i] + e
            assert s < e and e <= dti_block_thresh[i+1]

            part_debug_selection_idx = debug_permutation_idx[select_start:select_end].copy()
            start = np.min(part_debug_selection_idx[part_debug_selection_idx < main_size])
            end = np.max(part_debug_selection_idx[part_debug_selection_idx < main_size]) + np.int64(1)

            part_debug_selection_idx[part_debug_selection_idx < main_size] -= start
            part_debug_selection_idx[part_debug_selection_idx >= main_size] -= main_size - end + start


            property = np.concatenate([_prop(i, start - _dti_block_thresh[i], end - _dti_block_thresh[i]), debug_property])
            first_sub_blk = property[0, 3]
            property = property[part_debug_selection_idx]
            part_debug_sample_recover_idx = np.argsort(part_debug_selection_idx)[end-start:]

            sub_block_trace_idx = part_debug_sample_recover_idx.astype(np.int64).copy()

            while True:
                turn_idx = np.logical_and(np.isin(sub_block_trace_idx, part_debug_sample_recover_idx),
                                          sub_block_trace_idx >= 0).nonzero()[0]
                if turn_idx.shape[0] == 0:
                    break
                sub_block_trace_idx[turn_idx] -= 1

            property[part_debug_sample_recover_idx, 3] = first_sub_blk
            property[part_debug_sample_recover_idx[sub_block_trace_idx >= 0], 3] = \
                property[sub_block_trace_idx[sub_block_trace_idx >= 0], 3]

            return property

        return prop

    def add_debug_conn(_conn):
        def conn(i, s, e):
            select_start = dti_block_thresh[i] + s
            select_end = dti_block_thresh[i] + e
            assert s < e and e <= dti_block_thresh[i+1]

            part_selection_idx = debug_permutation_idx[select_start:select_end].copy()

            for di, (ds, de) in enumerate(zip(debug_block_thresh[:-1], debug_block_thresh[1:])):
                for output_neuron_idx, input_neuron_idx, input_channel_offset, value in debug_conn(di, ds, de):
                    output_neuron_idx = np.take(debug_recover_idx, output_neuron_idx + main_size, axis=0)
                    input_neuron_idx = np.take(debug_recover_idx, input_neuron_idx + main_size, axis=0)

                    _idx = np.logical_and(select_start <= output_neuron_idx, output_neuron_idx< select_end).nonzero()[0]

                    output_neuron_idx =  np.take(output_neuron_idx, _idx, axis=0)
                    input_neuron_idx =  np.take(input_neuron_idx, _idx, axis=0)
                    input_channel_offset =  np.take(input_channel_offset, _idx, axis=0)
                    value =  np.take(value, _idx, axis=0)
                    yield output_neuron_idx, input_neuron_idx, input_channel_offset, value

            start = np.min(part_selection_idx[part_selection_idx < main_size])
            end = np.max(part_selection_idx[part_selection_idx < main_size]) + np.int64(1)

            for output_neuron_idx, input_neuron_idx, input_channel_offset, value in \
                    _conn(i, start - _dti_block_thresh[i], end - _dti_block_thresh[i]):
                output_neuron_idx = np.take(debug_recover_idx, output_neuron_idx, axis=0)
                input_neuron_idx = np.take(debug_recover_idx, input_neuron_idx, axis=0)
                yield output_neuron_idx, input_neuron_idx, input_channel_offset, value

        return conn

    prop = add_debug_prop(prop)
    conn = add_debug_conn(conn)

    return debug_selection_idx, prop, conn, dti_block_thresh


def get_block_threshold(number, dti_block_thresh):
    if isinstance(number, int):
        _block_number = (dti_block_thresh[-1] - 1) // number + 1
        block_threshold = np.concatenate([np.arange(0, dti_block_thresh[-1], _block_number, dtype=np.int64),
                                          np.array([dti_block_thresh[-1]], dtype=np.int64)])
    elif isinstance(number, list):
        weight = np.array(number)
        _block_number = dti_block_thresh[-1]/np.sum(weight)*weight
        block_threshold = np.add.accumulate(_block_number).astype(np.int64)
        block_threshold = np.concatenate([np.array([0], dtype=np.int64),
                                          block_threshold])
        block_threshold[-1] = dti_block_thresh[-1]
    else:
        raise ValueError
    return block_threshold


def turn_to_block_idx(idx, block_threshold, turn_format=False):
    block_idx = np.searchsorted(block_threshold, idx, side='right') - 1
    if turn_format:
        block_idx = block_idx.astype(np.int16)
    neuron_idx = idx - block_threshold[block_idx]
    if turn_format:
        neuron_idx = neuron_idx.astype(np.uint32)
    return block_idx, neuron_idx


def merge_dti_distributation_block(orig_path, new_path, dtype="single", number=1, debug_block_path=None, output_degree=False, MPI_rank=None, only_load=False):
    if callable(orig_path):
        prop, conn, dti_block_thresh = orig_path()
    else:
        prop, conn, dti_block_thresh = connect_for_block(orig_path, dense=False)
    if isinstance(dtype, str):
        dtype = [dtype]

    if debug_block_path is not None:
        debug_selection_idx, prop, conn, dti_block_thresh = add_debug(debug_block_path, prop, conn, dti_block_thresh, new_path, only_load)
    else:
        debug_selection_idx = None

    block_threshold = get_block_threshold(number, dti_block_thresh)

    if debug_block_path is not None:
        assert debug_selection_idx is not None
        if not only_load:
            np.save(os.path.join(new_path, "debug_selection_idx"),
                    np.ascontiguousarray(
                        np.stack(list(turn_to_block_idx(debug_selection_idx, block_threshold)), axis=1)))
            np.save(os.path.join(new_path, "debug_subblk_id"),
                    np.searchsorted(dti_block_thresh, debug_selection_idx, "right") - 1)

    def _process(block_i):
        for d in dtype:
            _new_path = os.path.join(new_path, d)
            os.makedirs(_new_path, exist_ok=True)
            storage_path = os.path.join(_new_path, "block_{}".format(block_i))
            if not os.path.exists(storage_path+'.npz'):
                break
        else:
            print("passing processing", block_i)
            return
        print("in processing", block_i)
        block_start = block_threshold[block_i]
        block_end = block_threshold[block_i+1]

        dti_block_selection = []
        for j, (s, e) in enumerate(zip(dti_block_thresh[:-1], dti_block_thresh[1:])):
            if s >= block_start and e <= block_end:
                s1 = 0
                e1 = e - s
            elif s <= block_start and e >= block_end:
                s1 = block_start - s
                e1 = block_end - s
            elif s >= block_start and s < block_end:
                s1 = 0
                e1 = block_end - s
            elif e > block_start and e <= block_end:
                s1 = block_start - s
                e1 = e - s
            else:
                continue
            assert s1 >= 0 and e1 > s1 and e1 <= e - s
            dti_block_selection.append((j, s1, e1))
            print("property finished", j)

        _property = []
        for dti_i, s, e in dti_block_selection:
            _property.append(prop(dti_i, s, e))
        _property = np.concatenate(_property)
        assert _property.shape[0] == block_end - block_start

        _value = []
        _output_neuron_idx = []
        _input_neuron_idx = []
        _input_channel_offset = []
        for dti_i, s, e in dti_block_selection:
            for output_neuron_idx, input_neuron_idx, input_channel_offset, value in conn(dti_i, s, e):
                _value.append(value)
                _output_neuron_idx.append(output_neuron_idx)
                _input_neuron_idx.append(input_neuron_idx)
                _input_channel_offset.append(input_channel_offset)
        _output_neuron_idx = np.concatenate(_output_neuron_idx)
        _input_channel_offset = np.concatenate(_input_channel_offset)
        _input_neuron_idx = np.concatenate(_input_neuron_idx)
        _value = np.concatenate(_value)

        assert (np.unique(_output_neuron_idx) == np.arange(block_start, block_end, dtype=_output_neuron_idx.dtype)).all()

        _output_neuron_idx = (_output_neuron_idx - block_start).astype(np.uint32)
        _input_block_idx, _input_neuron_idx = turn_to_block_idx(_input_neuron_idx, block_threshold, turn_format=True)
        if not output_degree:
            new_weight_idx = np.lexsort((_input_channel_offset, _input_neuron_idx, _input_block_idx, _output_neuron_idx))
        else:
            new_weight_idx = np.lexsort((_input_channel_offset, _output_neuron_idx, _input_neuron_idx, _input_block_idx))

        _value = np.take(_value, new_weight_idx, axis=0)
        _output_neuron_idx = np.take(_output_neuron_idx, new_weight_idx, axis=0)
        _input_block_idx = np.take(_input_block_idx, new_weight_idx, axis=0)
        _input_block_idx -= block_i
        _input_neuron_idx = np.take(_input_neuron_idx, new_weight_idx, axis=0)
        _input_channel_offset = np.take(_input_channel_offset, new_weight_idx, axis=0)
        print("done", block_i)

        for d in dtype:
            _new_path = os.path.join(new_path, d)
            os.makedirs(_new_path, exist_ok=True)
            if d == "single":
                _value = _value
            elif d == "half":
                _value = _value.astype(np.float16)
            else:
                raise ValueError
            storage_path = os.path.join(_new_path, "block_{}".format(block_i))
            np.savez(storage_path,
                     property=_property,
                     output_neuron_idx=_output_neuron_idx,
                     input_block_idx=_input_block_idx,
                     input_neuron_idx=_input_neuron_idx,
                     input_channel_offset=_input_channel_offset,
                     weight=_value)

    block_numbers = block_threshold[1:] - block_threshold[:-1]
    assert (block_numbers > 0).all()

    if MPI_rank is None:
        with Thpool() as p:
            p.map(_process, range(0, block_numbers.shape[0]))
    else:
        assert 0 <= MPI_rank and MPI_rank < block_numbers.shape[0]
        _process(MPI_rank)
    return block_threshold

