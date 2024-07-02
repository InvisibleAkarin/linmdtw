import numpy as np
from .alignmenttools import get_diag_len, get_diag_indices, update_alignment_metadata
import warnings

def check_euclidean_inputs(X, Y):
    """
    检查两个欧几里得空间中的时间序列输入，它们将相互变形。它们必须满足以下条件：
    1. 它们在相同的维度空间中
    2. 它们是 32 位的
    3. 它们是 C 连续顺序的
    
    如果不满足条件 2 或 3，则自动修复并警告用户。
    此外，如果 X 或 Y 的列数多于行数，也会警告用户，
    因为惯例是点沿行排列，维度沿列排列。
    
    参数
    ----------
    X: ndarray(M, d)
        第一个时间序列
    Y: ndarray(N, d)
        第二个时间序列
    
    返回
    -------
    X: ndarray(M, d)
        第一个时间序列，可能在内存中被复制为 32 位，C 连续的
    Y: ndarray(N, d)
        第二个时间序列，可能在内存中被复制为 32 位，C 连续的
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("输入的时间序列不在相同的维度空间中")
    if X.shape[0] < X.shape[1]:
        warnings.warn("X {} 的列数多于行数；你是否想要转置？".format(X.shape))
    if Y.shape[0] < Y.shape[1]:
        warnings.warn("Y {} 的列数多于行数；你是否想要转置？".format(Y.shape))
    if not X.dtype == np.float32:
        warnings.warn("X 不是 32 位的，因此创建 32 位版本")
        X = np.array(X, dtype=np.float32)
    if not X.flags['C_CONTIGUOUS']:
        warnings.warn("X 不是 C 连续的；创建一个 C 连续的副本")
        X = X.copy(order='C')
    if not Y.dtype == np.float32:
        warnings.warn("Y 不是 32 位的，因此创建 32 位版本")
        Y = np.array(Y, dtype=np.float32)
    if not Y.flags['C_CONTIGUOUS']:
        warnings.warn("Y 不是 C 连续的；创建一个 C 连续的副本")
        Y = Y.copy(order='C')
    return X, Y

def dtw_brute(X, Y, debug=False):
    """
    计算两个时间有序的欧几里得点云之间的暴力动态时间规整，
    使用 cython 作为后端

    参数
    ----------
    X: ndarray(M, d)
        一个包含 M 个点的 d 维欧几里得点云
    Y: ndarray(N, d)
        一个包含 N 个点的 d 维欧几里得点云
    debug: boolean
        是否跟踪调试信息
    
    返回
    -------
    {
        'cost': float
            对齐的最优代价（如果计算没有提前停止），
        'U'/'L'/'UL': ndarray(M, N)
            选择矩阵（如果调试），
        'S': ndarray(M, N)
            累积代价矩阵（如果调试）
    }
    """
    from dynseqalign import DTW
    X, Y = check_euclidean_inputs(X, Y)
    return DTW(X, Y, int(debug))

def dtw_brute_backtrace(X, Y, debug=False):
    """
    计算两个时间有序的欧几里得点云之间的动态时间规整，
    使用 cython 作为后端。然后，通过回溯指针矩阵提取对齐路径

    参数
    ----------
    X: ndarray(M, d)
        一个包含 M 个点的 d 维欧几里得点云
    Y: ndarray(N, d)
        一个包含 N 个点的 d 维欧几里得点云
    debug: boolean
        是否跟踪调试信息
    
    返回
    -------
    如果不调试：
        (float: cost, ndarray(K, 2): 变形路径)
    
    如果调试：
    {
        'cost': float
            对齐的最优代价（如果计算没有提前停止），
        'U'/'L'/'UL': ndarray(M, N)
            选择矩阵（如果调试），
        'S': ndarray(M, N)
            累积代价矩阵（如果调试），
        'path': ndarray(K, 2)
            变形路径
    }
    """
    res = dtw_brute(X, Y, debug)
    res['P'] = np.asarray(res['P'])
    if debug: # pragma: no cover
        for key in ['U', 'L', 'UL', 'S']:
            res[key] = np.asarray(res[key])
    i = X.shape[0]-1
    j = Y.shape[0]-1
    path = [[i, j]]
    step = [[0, -1], [-1, 0], [-1, -1]] # 左，上，对角线
    while not(path[-1][0] == 0 and path[-1][1] == 0):
        s = step[res['P'][i, j]]
        i += s[0]
        j += s[1]
        path.append([i, j])
    path.reverse()
    path = np.array(path, dtype=int)
    if debug: # pragma: no cover
        res['path'] = path
        return res
    return (res['cost'], path)
    

def dtw_diag(X, Y, k_save = -1, k_stop = -1, box = None, reverse=False, debug=False, metadata=None):
    """
    线性内存对角 DTW 的 CPU 版本

    参数
    ----------
    X: ndarray(M, d)
        一个包含 M 个点的 d 维欧几里得点云
    Y: ndarray(N, d)
        一个包含 N 个点的 d 维欧几里得点云
    k_save: int
        保存 d0、d1 和 d2 的对角线 d2 的索引
    k_stop: int
        停止计算的对角线 d2 的索引
    box: list
        一个 [startx, endx, starty, endy] 的列表
    reverse: boolean
        是否反向计算
    debug: boolean
        是否保存累积代价矩阵
    metadata: dictionary
        用于存储计算信息的字典
    
    返回
    -------
    {
        'cost': float
            对齐的最优代价（如果计算没有提前停止），
        'U'/'L'/'UL': ndarray(M, N)
            选择矩阵（如果调试），
        'd0'/'d1'/'d2': ndarray(min(M, N))
            如果选择了保存索引，则保存的行，
        'csm0'/'csm1'/'csm2': ndarray(min(M, N))
            如果选择了保存索引，则保存的交叉相似度距离
    }
    """
    from dynseqalign import DTW_Diag_Step
    if not box:
        box = [0, X.shape[0]-1, 0, Y.shape[0]-1]
    M = box[1] - box[0] + 1
    N = box[3] - box[2] + 1
    box = np.array(box, dtype=np.int32)
    if k_stop == -1:
        k_stop = M+N-2
    if k_save == -1:
        k_save = k_stop

    # 调试信息
    U = np.zeros((1, 1), dtype=np.float32)
    L = np.zeros_like(U)
    UL = np.zeros_like(U)
    S = np.zeros_like(U)
    CSM = np.zeros_like(U)
    if debug: # pragma: no cover
        U = np.zeros((M, N), dtype=np.float32)
        L = np.zeros_like(U)
        UL = np.zeros_like(U)
        S = np.zeros_like(U)
        CSM = np.zeros_like(U)
    
    # 对角线
    diagLen = min(M, N)
    d0 = np.zeros(diagLen, dtype=np.float32)
    d1 = np.zeros_like(d0)
    d2 = np.zeros_like(d0)
    # 沿对角线的点之间的距离
    csm0 = np.zeros_like(d0)
    csm1 = np.zeros_like(d1)
    csm2 = np.zeros_like(d2)
    csm0len = diagLen
    csm1len = diagLen
    csm2len = diagLen

    # 遍历对角线
    res = {}
    for k in range(k_stop+1):
        DTW_Diag_Step(d0, d1, d2, csm0, csm1, csm2, X, Y, diagLen, box, int(reverse), k, int(debug), U, L, UL, S)
        csm2len = get_diag_len(box, k)
        if debug:
            i, j = get_diag_indices(M, N, k)
            CSM[i, j] = csm2[0:i.size]
        if metadata:
            update_alignment_metadata(metadata, csm2len)
        if k == k_save:
            res['d0'] = d0.copy()
            res['csm0'] = csm0.copy()
            res['d1'] = d1.copy()
            res['csm1'] = csm1.copy()
            res['d2'] = d2.copy()
            res['csm2'] = csm2.copy()
        if k < k_stop:
            # 移动对角线（三重缓冲）
            temp = d0
            d0 = d1
            d1 = d2
            d2 = temp
            temp = csm0
            csm0 = csm1
            csm1 = csm2
            csm2 = temp
            temp = csm0len
            csm0len = csm1len
            csm1len = csm2len
            csm2len = temp
    res['cost'] = d2[0] + csm2[0]
    if debug: # pragma: no cover
        res['U'] = U
        res['L'] = L
        res['UL'] = UL
        res['S'] = S
        res['CSM'] = CSM
    return res

def linmdtw(X, Y, box=None, min_dim=500, do_gpu=True, metadata=None):
    """
    线性内存精确、可并行化的 DTW

    参数
    ----------
    X: ndarray(N1, d)
        一个包含 N1 个点的 d 维欧几里得点云
    Y: ndarray(N2, d)
        一个包含 N2 个点的 d 维欧几里得点云
    min_dim: int
        如果矩形区域的左侧或右侧的维度之一小于此数值，
        则切换到暴力 CPU 计算
    do_gpu: boolean
        如果为 True，使用 GPU 对角 DTW 函数作为子程序。
        否则，使用 CPU 版本。两者都是线性内存，但
        对于较大的同步问题，GPU 会更快
    metadata: dictionary
        用于存储计算信息的字典
    
    返回
    -------
        (float: cost, ndarray(K, 2): 最优变形路径)
    """
    X, Y = check_euclidean_inputs(X, Y)
    dtw_diag_fn = dtw_diag
    if do_gpu:
        from .dtwgpu import DTW_GPU_Initialized, init_gpu, dtw_diag_gpu
        if not DTW_GPU_Initialized:
            init_gpu()
        from .dtwgpu import DTW_GPU_Failed
        if DTW_GPU_Failed:
            warnings.warn("回退到 CPU")
            do_gpu = False
        else:
            dtw_diag_fn = dtw_diag_gpu
    if not box:
        box = [0, X.shape[0]-1, 0, Y.shape[0]-1]
    M = box[1]-box[0]+1
    N = box[3]-box[2]+1

    # 停止条件，回退到CPU
    if M < min_dim or N < min_dim:
        if metadata:
            metadata['totalCells'] += M*N
        cost, path = dtw_brute_backtrace(X[box[0]:box[1]+1, :], Y[box[2]:box[3]+1, :])
        for p in path:
            p[0] += box[0]
            p[1] += box[2]
        return (cost, path)
    
    # 否则，继续递归
    K = M + N - 1
    # 进行前向计算
    k_save = int(np.ceil(K/2.0))
    res1 = dtw_diag_fn(X, Y, k_save=k_save, k_stop=k_save, box=box, metadata=metadata)

    # 进行后向计算
    k_save_rev = k_save
    if K%2 == 0:
        k_save_rev += 1
    res2 = dtw_diag_fn(X, Y, k_save=k_save_rev, k_stop=k_save_rev, box=box, reverse=True, metadata=metadata)
    res2['d0'], res2['d2'] = res2['d2'], res2['d0']
    res2['csm0'], res2['csm2'] = res2['csm2'], res2['csm0']
    # 去掉多余的对角元素
    for i in range(3):
        sz = get_diag_len(box, k_save-2+i)
        res1['d%i'%i] = res1['d%i'%i][0:sz]
        res1['csm%i'%i] = res1['csm%i'%i][0:sz]
    for i in range(3):
        sz = get_diag_len(box, k_save_rev-2+i)
        res2['d%i'%i] = res2['d%i'%i][0:sz]
        res2['csm%i'%i] = res2['csm%i'%i][0:sz]
    # 对齐反向对角线
    for d in ['d0', 'd1', 'd2', 'csm0', 'csm1', 'csm2']:
        res2[d] = res2[d][::-1]
    
    # 在三个对角线中找到最小成本并在该元素上拆分
    min_cost = np.inf
    min_idxs = []
    for k in range(3):
        dleft = res1['d%i'%k]
        dright = res2['d%i'%k]
        csmright = res2['csm%i'%k]
        diagsum = dleft + dright + csmright
        idx = np.argmin(diagsum)
        if diagsum[idx] < min_cost:
            min_cost = diagsum[idx]
            i, j = get_diag_indices(X.shape[0], Y.shape[0], k_save-2+k, box)
            min_idxs = [i[idx], j[idx]]

    # 递归计算左路径
    left_path = []
    box_left = [box[0], min_idxs[0], box[2], min_idxs[1]]
    left_path = linmdtw(X, Y, box_left, min_dim, do_gpu, metadata)[1]

    # 递归计算右路径
    right_path = []
    box_right = [min_idxs[0], box[1], min_idxs[1], box[3]]
    right_path = linmdtw(X, Y, box_right, min_dim, do_gpu, metadata)[1]
    
    return (min_cost, np.concatenate((left_path, right_path[1::, :]), axis=0))

    