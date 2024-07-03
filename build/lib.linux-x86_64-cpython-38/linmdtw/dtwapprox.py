import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import sparse
import time
import dynseqalign
from .dtw import dtw_brute_backtrace, linmdtw, check_euclidean_inputs
from .alignmenttools import get_path_cost

def fill_block(A, p, radius, val):
    """
    用值填充一个方块

    参数
    ----------
    A: ndarray(M, N) 或 sparse(M, N)
        要填充的数组
    p: list of [i, j]
        方块中心的坐标
    radius: int
        方块宽度的一半
    val: float
        要填充的值
    """
    i1 = max(p[0]-radius, 0)
    i2 = min(p[0]+radius, A.shape[0]-1)
    j1 = max(p[1]-radius, 0)
    j2 = min(p[1]+radius, A.shape[1]-1)
    A[i1:i2+1, j1:j2+1] = val

def _dtw_constrained_occ(X, Y, Occ, debug=False, level=0, do_plot=False):
    """
    在受限占用掩码上进行 DTW。一个用于 fastdtw 和 cdtw 的辅助方法
    
    参数
    ----------
    X: ndarray(M, d)
        一个包含 M 个点的 d 维欧几里得点云
    Y: ndarray(N, d)
        一个包含 N 个点的 d 维欧几里得点云
    Occ: scipy.sparse((M, N))
        一个 MxN 的数组，如果这个单元格要被评估则为 1，否则为 0
    debug: boolean
        是否跟踪调试信息
    level: int
        用于跟踪递归层级的整数（如果适用）
    do_plot: boolean
        是否在每个层级绘制变形路径并保存为图像文件
    
    返回
    -------
        (float: cost, ndarray(K, 2): 变形路径)
    """
    M = X.shape[0]
    N = Y.shape[0]
    tic = time.time()
    S = sparse.lil_matrix((M, N))
    P = sparse.lil_matrix((M, N), dtype=int)
    I, J = Occ.nonzero()
    # 按栅格顺序排序单元格
    idx = np.argsort(J)
    I = I[idx]
    J = J[idx]
    idx = np.argsort(I, kind='stable')
    I = I[idx]
    J = J[idx]

    ## 步骤2：找到左、上和对角邻居的索引。
    # 所有邻居必须在边界内，并且在稀疏结构内
    # 使idx为M+1 x N+1，这样-1将环绕到0
    # 使其为1索引，这样所有有效条目索引都大于0
    idx = sparse.coo_matrix((np.arange(I.size)+1, (I, J)), shape=(M+1, N+1)).tocsr()
    # 左邻居
    left = np.array(idx[I, J-1], dtype=np.int32).flatten()
    left[left <= 0] = -1
    left -= 1
    # 上邻居
    up = np.array(idx[I-1, J], dtype=np.int32).flatten()
    up[up <= 0] = -1
    up -= 1
    # 对角邻居
    diag = np.array(idx[I-1, J-1], dtype=np.int32).flatten()
    diag[diag <= 0] = -1
    diag -= 1

    ## 步骤3：将信息传递给cython进行动态规划步骤
    S = np.zeros(I.size, dtype=np.float32) # 动态规划矩阵
    P = np.zeros(I.size, dtype=np.int32) # 路径指针矩阵
    dynseqalign.FastDTW_DynProg_Step(X, Y, I, J, left, up, diag, S, P)
    P = sparse.coo_matrix((P, (I, J)), shape=(M, N)).tocsr()
    if debug or do_plot: # pragma: no cover
        S = sparse.coo_matrix((S, (I, J)), shape=(M, N)).tocsr()
    
    # 步骤4：进行回溯
    i = M-1
    j = N-1
    path = [[i, j]]
    step = [[0, -1], [-1, 0], [-1, -1]] # 左, 上, 对角
    while not(path[-1][0] == 0 and path[-1][1] == 0):
        s = step[P[i, j]]
        i += s[0]
        j += s[1]
        path.append([i, j])
    path.reverse()
    path = np.array(path, dtype=int)
    
    if do_plot: # pragma: no cover
        plt.figure(figsize=(8, 8))
        plt.imshow(S.toarray())
        path = np.array(path)
        plt.scatter(path[:, 1], path[:, 0], c='C1')
        plt.title("Level {}".format(level))
        plt.savefig("%i.png"%level, bbox_inches='tight')

    if debug: # pragma: no cover
        return {'path':path, 'S':S, 'P':P}
    else:
        return (get_path_cost(X, Y, path), path)

def fastdtw(X, Y, radius, debug=False, level = 0, do_plot=False):
    """
    实现 [1]
    [1] FastDTW: Toward Accurate Dynamic Time Warping in Linear Time and Space. Stan Salvador 和 Philip Chan
    
    参数
    ----------
    X: ndarray(M, d)
        一个包含 M 个点的 d 维欧几里得点云
    Y: ndarray(N, d)
        一个包含 N 个点的 d 维欧几里得点云
    radius: int
        l-无穷大盒子的半径，决定了每个层级的稀疏结构
    debug: boolean
        是否跟踪调试信息
    level: int
        用于跟踪递归层级的整数
    do_plot: boolean
        是否在每个层级绘制变形路径并保存为图像文件
    
    返回
    -------
        (float: cost, ndarray(K, 2): 变形路径)
    """
    X, Y = check_euclidean_inputs(X, Y)
    minTSsize = radius + 2
    M = X.shape[0]
    N = Y.shape[0]
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)
    if M < radius or N < radius:
        return dtw_brute_backtrace(X, Y)
    # 递归步骤
    cost, path = fastdtw(X[0::2, :], Y[0::2, :], radius, debug, level+1, do_plot)
    if type(path) is dict:
        path = path['path']
    path = np.array(path)
    path *= 2
    Occ = sparse.lil_matrix((M, N))

    ## 步骤1：找出被占用单元格的索引
    for p in path:
        fill_block(Occ, p, radius, 1)
    return _dtw_constrained_occ(X, Y, Occ, debug, level, do_plot)

def cdtw(X, Y, radius, debug=False, do_plot=False):
    """
    带约束的动态时间规整，按照 Sakoe-Chiba 带
    
    参数
    ----------
    X: ndarray(M, d)
        一个包含 M 个点的 d 维欧几里得点云
    Y: ndarray(N, d)
        一个包含 N 个点的 d 维欧几里得点云
    radius: int
        变形路径允许偏离恒等映射的最大距离
    debug: boolean
        是否跟踪调试信息
    do_plot: boolean
        是否在每个层级绘制变形路径并保存为图像文件
    
    返回
    -------
        (float: cost, ndarray(K, 2): 变形路径)
    """
    radius = int(max(radius, 1))
    X, Y = check_euclidean_inputs(X, Y)
    M = X.shape[0]
    N = Y.shape[0]
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)
    if M < radius or N < radius:
        return dtw_brute_backtrace(X, Y)
    ## 步骤1：找出被占用单元格的索引
    Occ = sparse.lil_matrix((M, N))
    slope = M/N
    for i in range(max(M, N)):
        p = []
        if M < N:
            p = [int(slope*i), i]
        else:
            p = [i, int(i/slope)]
        fill_block(Occ, p, radius, 1)
    return _dtw_constrained_occ(X, Y, Occ, debug, 0, do_plot)
    


def get_box_area(a1, a2):
    """
    获取由两个锚点指定的方块面积
    
    参数
    ----------
    a1: list(2)
        第一个锚点的行/列
    a2: list(2)
        第二个锚点的行/列
    
    返回
    -------
    由这两个锚点确定的方块面积
    """
    m = a2[0]-a1[0]+1
    n = a2[1]-a1[1]+1
    return m*n

def mrmsdtw(X, Y, tau, debug=False, refine=True):
    """
    实现了 [2] 中的近似、内存受限的多尺度 DTW 技术
    [2] "Memory-Restricted Multiscale Dynamic Time Warping"
    Thomas Praetzlich, Jonathan Driedger 和 Meinard Mueller
    
    参数
    ----------
    X: ndarray(M, d)
        一个包含 M 个点的 d 维欧几里得点云
    Y: ndarray(N, d)
        一个包含 N 个点的 d 维欧几里得点云
    tau: int
        任意时刻在内存中的最大单元格数量
    debug: boolean
        是否跟踪调试信息
    refine: boolean
        是否使用“白色锚点”进行细化
    
    返回
    -------
    path: ndarray(K, 2)
        变形路径
    """
    X, Y = check_euclidean_inputs(X, Y)
    M = X.shape[0]
    N = Y.shape[0]
    if M*N < tau:
        # 如果矩阵已经在内存限制范围内，则直接返回DTW
        return dtw_brute_backtrace(X, Y)

    ## 步骤1：在粗略级别执行DTW
    # 根据内存需求确定粗略对齐的子采样因子
    d = int(np.ceil(np.sqrt(M*N/tau)))
    X2 = np.ascontiguousarray(X[0::d, :])
    Y2 = np.ascontiguousarray(Y[0::d, :])
    anchors = dtw_brute_backtrace(X2, Y2)[1]
    anchors = (np.array(anchors)*d).tolist()
    if anchors[-1][0] < M-1 or anchors[-1][1] < N-1:
        anchors.append([M-1, N-1])
    
    ## 步骤2：如有必要，细分锚点以保持在内存限制范围内
    idx = 0
    while idx < len(anchors)-1:
        a1 = anchors[idx]
        a2 = anchors[idx+1]
        if get_box_area(a1, a2) > tau:
            # 细分单元格
            i = int((a1[0]+a2[0])/2)
            j = int((a1[1]+a2[1])/2)
            anchors = anchors[0:idx+1] + [[i, j]] + anchors[idx+1::]
        else:
            # 继续
            idx += 1
    
    ## 步骤3：在每个块中进行对齐
    path = np.array([], dtype=int)
    # 记录路径中的“黑色锚点”索引
    banchors_idx = [0]
    for i in range(len(anchors)-1):
        a1 = anchors[i]
        a2 = anchors[i+1]
        box = [a1[0], a2[0], a1[1], a2[1]]
        pathi = linmdtw(X, Y, box=box)[1]
        if path.size == 0:
            path = pathi
        else:
            path = np.concatenate((path, pathi[0:-1, :]), axis=0)
        banchors_idx.append(len(path)-1)
    # 添加最后的端点
    path = np.concatenate((path, np.array([[M-1, N-1]], dtype=int)), axis=0)
    if not refine:
        return (get_path_cost(X, Y, path), path)
    
    ## 步骤4：提出“白色锚点”集合
    # 首先选择它们位于每个块的中心
    wanchors_idx = []
    for idx in range(len(banchors_idx)-1):
        wanchors_idx.append([int(0.5*(banchors_idx[idx]+banchors_idx[idx+1]))]*2)
    # 如果块太大，则拆分锚点位置
    for i in range(len(wanchors_idx)-1):
        a1 = path[wanchors_idx[i][-1]]
        a2 = path[wanchors_idx[i+1][0]]
        while get_box_area(a1, a2) > tau:
            # 将锚点相互移动
            wanchors_idx[i][-1] += 1
            wanchors_idx[i+1][0] -= 1
            a1 = path[wanchors_idx[i][-1]]
            a2 = path[wanchors_idx[i+1][0]]
    ## 步骤5：在白色锚点之间执行DTW并拼接路径
    pathret = path[0:wanchors_idx[0][0]+1, :]
    for i in range(len(wanchors_idx)-1):
        a1 = path[wanchors_idx[i][-1]]
        a2 = path[wanchors_idx[i+1][0]]
        box = [a1[0], a2[0], a1[1], a2[1]]
        pathi = linmdtw(X, Y, box=box)[1]
        pathret = np.concatenate((pathret, pathi[0:-1, :]), axis=0)
        # 如果这个框和下一个框之间有间隙，使用之前的路径
        i1 = wanchors_idx[i+1][0]
        i2 = wanchors_idx[i+1][1]
        if i1 != i2:
            pathret = np.concatenate((pathret, path[i1:i2, :]), axis=0)
    i1 = wanchors_idx[-1][-1]
    pathret = np.concatenate((pathret, path[i1::, :]), axis=0)
    return (get_path_cost(X, Y, pathret), pathret)