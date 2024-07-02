import numpy as np
from numba import jit

def get_diag_len(box, k):
    """
    返回特定对角线中的元素数量

    参数
    ----------
    MTotal: int
        X 中样本的总数
    NTotal: int
        Y 中样本的总数
    k: int
        对角线的索引
    
    返回
    -------
    元素数量
    """
    M = box[1] - box[0] + 1
    N = box[3] - box[2] + 1
    starti = k
    startj = 0
    if k >= M:
        starti = M-1
        startj = k - (M-1)
    endj = k
    endi = 0
    if k >= N:
        endj = N-1
        endi = k - (N-1)
    return endj-startj+1


def get_diag_indices(MTotal, NTotal, k, box = None, reverse=False):
    """
    计算对角线上的索引到累积距离矩阵中的索引

    参数
    ----------
    MTotal: int
        X 中样本的总数
    NTotal: int
        Y 中样本的总数
    k: int
        对角线的索引
    box: list [XStart, XEnd, YStart, YEnd]
        搜索范围的坐标
    reverse: boolean
        是否反向计算索引
    
    返回
    -------
    i: ndarray(dim)
        行索引
    j: ndarray(dim)
        列索引
    """
    if not box:
        box = [0, MTotal-1, 0, NTotal-1]
    M = box[1] - box[0] + 1
    N = box[3] - box[2] + 1
    starti = k
    startj = 0
    if k > M-1:
        starti = M-1
        startj = k - (M-1)
    i = np.arange(starti, -1, -1)
    j = startj + np.arange(i.size)
    dim = np.sum(j < N) # 该对角线的长度
    i = i[0:dim]
    j = j[0:dim]
    if reverse:
        i = M-1-i
        j = N-1-j
    i += box[0]
    j += box[2]
    return i, j

def update_alignment_metadata(metadata = None, newcells = 0):
    """
    将新处理的单元格数量添加到总处理单元格数量中，
    并在有进展时打印出一个百分比点

    参数
    ----------
    newcells: int
        新处理的单元格数量
    metadata: dictionary
        包含 'M', 'N', 'totalCells'（均为整数）和 'timeStart' 的字典
    """
    if metadata:
        if 'M' in metadata and 'N' in metadata and 'totalCells' in metadata:
            import time
            denom = metadata['M']*metadata['N']
            before = np.floor(50*metadata['totalCells']/denom)
            metadata['totalCells'] += newcells
            after = np.floor(50*metadata['totalCells']/denom)
            perc = 1
            if 'perc' in metadata:
                perc = metadata['perc']
            if after > before and after % perc == 0:
                print("Parallel Alignment {}% ".format(after), end='')
                if 'timeStart' in metadata:
                    print("Elapsed time: %.3g"%(time.time()-metadata['timeStart']))

def get_csm(X, Y): # pragma: no cover
    """
    返回 X 和 Y 之间的欧几里得交叉相似性矩阵

    参数
    ---------
    X: ndarray(M, d)
        一个包含 M 个点的 d 维欧几里得点云
    Y: ndarray(N, d)
        一个包含 N 个点的 d 维欧几里得点云
    
    返回
    -------
    D: ndarray(M, N)
        交叉相似性矩阵
    
    """
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    C = XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def get_ssm(X): # pragma: no cover
    """
    返回时间有序的欧几里得点云 X 的自相似矩阵

    参数
    ---------
    X: ndarray(M, d)
        一个包含 M 个点的 d 维欧几里得点云
    
    返回
    -------
    D: ndarray(M, M)
        自相似矩阵
    """
    return get_csm(X, X)

def get_path_cost(X, Y, path):
    """
    返回匹配两个欧几里得点云的变形路径的代价

    参数
    ---------
    X: ndarray(M, d)
        一个包含 M 个点的 d 维欧几里得点云
    Y: ndarray(N, d)
        一个包含 N 个点的 d 维欧几里得点云
    path: ndarray(K, 2)
        变形路径
    
    返回
    -------
    cost: float
        变形路径上 X 和 Y 之间欧几里得距离的总和
    """
    x = X[path[:, 0], :]
    y = Y[path[:, 1], :]
    ds = np.sqrt(np.sum((x-y)**2, 1))
    return np.sum(ds)

def make_path_strictly_increase(path): # pragma: no cover
    """
    给定一个变形路径，移除所有不严格递增的行
    """
    toKeep = np.ones(path.shape[0])
    i0 = 0
    for i in range(1, path.shape[0]):
        if np.abs(path[i0, 0] - path[i, 0]) >= 1 and np.abs(path[i0, 1] - path[i, 1]) >= 1:
            i0 = i
        else:
            toKeep[i] = 0
    return path[toKeep == 1, :]

def refine_warping_path(path):
    """
    实现了 Ewert 和 Müller 在 "Refinement Strategies for Music Synchronization" 第 4 节中的技术

    参数
    ----------
    path: ndarray(K, 2)
        一个变形路径
    
    返回
    -------
    path_refined: ndarray(N >= K, 2)
        一组精炼的对应关系
    """
    N = path.shape[0]
    ## 步骤1：识别所有垂直和水平线段
    vert_horiz = []
    i = 0
    while i < N-1:
        if path[i+1, 0] == path[i, 0]:
            # 垂直线
            j = i+1
            while path[j, 0] == path[i, 0] and j < N-1:
                j += 1
            if j < N-1:
                vert_horiz.append({'type':'vert', 'i':i, 'j':j-1})
                i = j-1
            else:
                vert_horiz.append({'type':'vert', 'i':i, 'j':j})
                i = j
        elif path[i+1, 1] == path[i, 1]:
            # 水平线
            j = i+1
            while path[j, 1] == path[i, 1] and j < N-1:
                j += 1
            if j < N-1:
                vert_horiz.append({'type':'horiz', 'i':i, 'j':j-1})
                i = j-1
            else:
                vert_horiz.append({'type':'horiz', 'i':i, 'j':j})
                i = j
        else:
            i += 1
    
    ## 步骤2：计算局部密度
    xidx = []
    density = []
    i = 0
    vhi = 0
    while i < N:
        inext = i+1
        if vhi < len(vert_horiz) and vert_horiz[vhi]['i'] == i: # 这是一个垂直或水平线段
            v = vert_horiz[vhi]
            n_seg = v['j']-v['i']+1
            xidxi = []
            densityi = []
            n_seg_prev = 0
            n_seg_next = 0
            if vhi > 0:
                v2 = vert_horiz[vhi-1]
                if i == v2['j']:
                    # 第一个线段在一个角落
                    n_seg_prev = v2['j']-v2['i']+1
            if vhi < len(vert_horiz) - 1:
                v2 = vert_horiz[vhi+1]
                if v['j'] == v2['i']:
                    # 最后一个线段是一个角落
                    n_seg_next = v2['j']-v2['i']+1
            # 情况1：垂直线段
            if v['type'] == 'vert':
                xidxi = [path[i, 0] + k/n_seg for k in range(n_seg+1)]
                densityi = [n_seg]*(n_seg+1)
                if n_seg_prev > 0:
                    densityi[0] = n_seg/n_seg_prev
                if n_seg_next > 0:
                    densityi[-2] = n_seg/n_seg_next
                    densityi[-1] = n_seg/n_seg_next
                    inext = v['j']
                else:
                    inext = v['j']+1
            # 情况2：水平线段
            else:  
                xidxi = [path[i, 0] + k for k in range(n_seg)]
                densityi = [1/n_seg]*n_seg
                if n_seg_prev > 0:
                    xidxi = xidxi[1::]
                    densityi = densityi[1::]
                if n_seg_next > 0:
                    inext = v['j']
                else:
                    inext = v['j']+1
            xidx += xidxi
            density += densityi
            vhi += 1
        else:
            # 这是一个对角线段
            xidx += [path[i, 0], path[i, 0]+1]
            density += [1, 1]
        i = inext
    
    ## 步骤3：整合密度
    xidx = np.array(xidx)
    density = np.array(density)
    path_refined = [[0, 0]]
    j = 0
    for i in range(1, xidx.size):
        if xidx[i] > xidx[i-1]:
            j += (xidx[i]-xidx[i-1])*density[i-1]
            path_refined.append([xidx[i], j])
    path_refined = np.array(path_refined)
    return path_refined

def get_alignment_area_dist(P1, P2, do_plot = False):
    """
    计算两个变形路径之间的基于面积的对齐误差。

    参数
    ----------
    ndarray(M, 2): P1
        第一个变形路径
    ndarray(N, 2): P2
        第二个变形路径
    do_plot: boolean
        是否绘制显示封闭区域的图
    
    返回
    -------
    score: float
        面积得分
    """
    import scipy.sparse as sparse
    M = np.max(P1[:, 0])
    N = np.max(P1[:, 1])
    A1 = sparse.lil_matrix((M, N))
    A2 = sparse.lil_matrix((M, N))
    for i in range(P1.shape[0]):
        [ii, jj] = [P1[i, 0], P1[i, 1]]
        [ii, jj] = [min(ii, M-1), min(jj, N-1)]
        A1[ii, jj::] = 1.0
    for i in range(P2.shape[0]):
        [ii, jj] = [P2[i, 0], P2[i, 1]]
        [ii, jj] = [min(ii, M-1), min(jj, N-1)]
        A2[ii, jj::] = 1.0
    A = np.abs(A1 - A2)
    dist = np.sum(A)/(M + N)
    if do_plot: # pragma: no cover
        import matplotlib.pyplot as plt
        plt.imshow(A.toarray())
        plt.scatter(P1[:, 1], P1[:, 0], 5, 'c', edgecolor = 'none')
        plt.scatter(P2[:, 1], P2[:, 0], 5, 'r', edgecolor = 'none')
        plt.title("Dist = %g"%dist)
    return dist


@jit(nopython=True)
def get_alignment_row_dists(P1, P2):
    """
    测量两个变形路径之间的误差。
    对于第一个路径中的每个点，记录在第二个路径中同一行上最近点的距离

    参数
    ----------
    P1: ndarray(M, 2)
        基准变形路径
    P2: ndarray(N, 2)
        测试变形路径
    
    返回
    -------
    dists: ndarray(M)
        第一个变形路径上每个点的误差
    """
    k = 0
    dists = np.zeros(P1.shape[0])
    i2 = 0
    for i1 in range(P1.shape[0]):
        # 沿着 P2 移动直到它在同一行
        while P2[i2, 0] != P1[i1, 0]:
            i2 += 1
        # 检查 P2 中所有具有相同行的条目
        mindist = abs(P2[i2, 1] - P1[i1, 1])
        k = i2+1
        while k < P2.shape[0] and P2[k, 0] == P1[i1, 0]:
            mindist = min(mindist, abs(P2[k, 1]-P1[i1, 1]))
            k += 1
        dists[i1] = mindist
    return dists

def get_alignment_row_col_dists(P1, P2):
    """
    测量两个变形路径之间的误差。
    对于第一个路径中的每个点，记录在第二个路径中同一行上最近点的距离，
    反之亦然。然后沿列重复此操作

    参数
    ----------
    P1: ndarray(M, 2)
        基准变形路径
    P2: ndarray(N, 2)
        测试变形路径
    
    返回
    -------
    dists: ndarray(2M+2N)
        误差
    """
    dists11 = get_alignment_row_dists(P1, P2)
    dists12 = get_alignment_row_dists(P2, P1)
    dists21 = get_alignment_row_dists(np.fliplr(P1), np.fliplr(P2))
    dists22 = get_alignment_row_dists(np.fliplr(P2), np.fliplr(P1))
    return np.concatenate((dists11, dists12, dists21, dists22))


def get_interpolated_euclidean_timeseries(X, t, kind='linear'):
    """
    使用 interp2d 对欧几里得空间中的时间序列进行重采样

    参数
    ----------
    X: ndarray(M, d)
        包含 n 个点的欧几里得时间序列
    t: ndarray(N)
        单位区间 [0, 1] 上的重新参数化函数
    kind: string
        插值的类型
    
    返回
    -------
    Y: ndarray(N, d)
        插值后的时间序列
    """
    import scipy.interpolate as interp
    M = X.shape[0]
    d = X.shape[1]
    t0 = np.linspace(0, 1, M)
    dix = np.arange(d)
    f = interp.interp2d(dix, t0, X, kind=kind)
    Y = f(dix, t)
    return Y

def get_inverse_fn_equally_sampled(t, x):
    """
    计算一维函数的反函数并对其进行等距采样。

    参数
    ---------
    t: ndarray(N)
        原函数的定义域样本
    x: ndarray(N)
        原函数的值域样本
    
    返回
    -------
    y: ndarray(N)
        反函数的采样值
    """
    import scipy.interpolate as interp
    N = len(t)
    t2 = np.linspace(np.min(x), np.max(x), N)
    try:
        res = interp.spline(x, t, t2)
        return res
    except:
        return t

def get_parameterization_dict(N, do_plot = False):
    """
    构建单位区间上不同类型参数化的字典

    参数
    ----------
    N: int
        单位区间上的样本数量
    do_plot: boolean
        是否绘制所有参数化的图
    
    返回
    -------
    D: ndarray(N, K)
        参数化字典，每个变形路径在不同的列中
    """
    t = np.linspace(0, 1, N)
    D = []
    # 多项式
    if do_plot: # pragma: no cover
        import matplotlib.pyplot as plt
        plt.subplot(131)
        plt.title('多项式')
    for p in range(-4, 6):
        tp = p
        if tp < 0:
            tp = -1.0/tp
        x = t**(tp**1)
        D.append(x)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.plot(x)
    # 指数/对数
    if do_plot: # pragma: no cover
        import matplotlib.pyplot as plt
        plt.subplot(132)
        plt.title('指数/对数')
    for p in range(2, 6):
        t = np.linspace(1, p**p, N)
        x = np.log(t)
        x = x - np.min(x)
        x = x/np.max(x)
        t = t/np.max(t)
        x2 = get_inverse_fn_equally_sampled(t, x)
        x2 = x2 - np.min(x2)
        x2 = x2/np.max(x2)
        D.append(x)
        D.append(x2)
        if do_plot: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.plot(x)
            plt.plot(x2)
    # 双曲正切
    if do_plot: # pragma: no cover
        import matplotlib.pyplot as plt
        plt.subplot(133)
        plt.title('双曲正切')
    for p in range(2, 5):
        t = np.linspace(-2, p, N)
        x = np.tanh(t)
        x = x - np.min(x)
        x = x/np.max(x)
        t = t/np.max(t)
        x2 = get_inverse_fn_equally_sampled(t, x)
        x2 = x2 - np.min(x2)
        x2 = x2/np.max(x2)
        D.append(x)
        D.append(x2)
        if do_plot: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.plot(x)
            plt.plot(x2)
    D = np.array(D)
    return D

def sample_parameterization_dict(D, k, do_plot = False): 
    """
    返回一个由 k 个随机元素组成的变形路径，
    这些元素从字典 D 中抽取

    参数
    ----------
    D: ndarray(N, K)
        变形路径的字典，每个变形路径在不同的列中
    k: int
        要组合的变形路径数量
    """
    N = D.shape[0]
    dim = D.shape[1]
    idxs = np.random.permutation(N)[0:k]
    weights = np.zeros(N)
    weights[idxs] = np.random.rand(k)
    weights = weights / np.sum(weights)
    res = weights.dot(D)
    res = res - np.min(res)
    res = res/np.max(res)
    if do_plot: # pragma: no cover
        import matplotlib.pyplot as plt
        plt.plot(res)
        for idx in idxs:
            plt.plot(np.arange(dim), D[idx, :], linestyle='--')
        plt.title('构建的扭曲路径')
    return res

def param_to_warppath(t, M):
    """
    将参数化函数转换为有效的变形路径

    参数
    ----------
    t: ndarray(N)
        单位区间上参数化函数的样本
    M: int
        原始时间序列中的样本数量
    
    返回
    -------
    P: ndarray(K, 2)
        最符合参数化的变形路径
    """
    N = len(t)
    P = np.zeros((N, 2), dtype=int)
    P[:, 0] = t*(M-1)
    P[:, 1] = np.arange(N)
    i = 0
    while i < P.shape[0]-1:
        diff = P[i+1, 0] - P[i, 0]
        if diff > 1:
            newchunk = np.zeros((diff-1, 2), dtype=int)
            newchunk[:, 1] = P[i, 1]
            newchunk[:, 0] = P[i, 0] + 1 + np.arange(diff-1)
            P = np.concatenate((P[0:i+1, :], newchunk, P[i+1::, :]), axis=0)
        i += 1
    return P