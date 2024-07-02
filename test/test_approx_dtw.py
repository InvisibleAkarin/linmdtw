import pytest
import numpy as np
import linmdtw

def get_pcs(N):
    t1 = np.linspace(0, 1, N)
    t2 = t1**2
    X = np.zeros((N, 2), dtype=np.float32)
    X[:, 0] = np.cos(2*np.pi*t1)
    X[:, 1] = np.sin(4*np.pi*t1)
    Y = np.zeros_like(X)
    Y[:, 0] = np.cos(2*np.pi*t2)
    Y[:, 1] = np.sin(4*np.pi*t2)
    return X, Y

class TestDTWApprox:

    def test_fasdtw(self):
        """
        表明 fastdtw 误差与搜索半径成反比
        """
        X, Y = get_pcs(1000)
        path1 = np.array(linmdtw.dtw_brute_backtrace(X, Y)[1])
        path2 = np.array(linmdtw.fastdtw(X, Y, 50)[1])
        path3 = np.array(linmdtw.fastdtw(X, Y, 5)[1])
        err1 = linmdtw.get_alignment_row_col_dists(path1, path2)
        err2 = linmdtw.get_alignment_row_col_dists(path1, path3)
        assert(np.mean(err1) <= np.mean(err2))
    
    def test_cdtw(self):
        """
        测试使用 Sakoe-Chiba 带的约束动态时间规整
        """
        np.random.seed(1)
        M = 100
        N = 150
        t1 = np.linspace(0, 1, M)
        X = np.zeros((M, 2), dtype=np.float32)
        X[:, 0] = np.cos(2*np.pi*t1)
        X[:, 1] = np.sin(8*np.pi*t1)
        ## 从参数化字典中抽取一个元素
        ## 并使用此参数化来插值原始时间序列
        D = linmdtw.alignmenttools.get_parameterization_dict(N)
        s = linmdtw.alignmenttools.sample_parameterization_dict(D, 4)
        Y = linmdtw.alignmenttools.get_interpolated_euclidean_timeseries(X, s)

        cost10, _ = linmdtw.cdtw(X, Y, 10)
        cost10_T, _ = linmdtw.cdtw(Y, X, 10)
        assert(cost10 == cost10_T)
        cost4, _ = linmdtw.cdtw(X, Y, 4)
        cost4_T, _ = linmdtw.cdtw(Y, X, 4)
        assert(cost4 == cost4_T)
        assert(cost10 < cost4)
        assert(cost10_T < cost4_T)
        

    def test_mrmsdtw(self):
        """
        测试当内存减少时，mrmsdtw 的误差是单调增加的
        """
        X, Y = get_pcs(1000)
        path1 = linmdtw.dtw_brute_backtrace(X, Y)[1]
        path1 = np.array(path1)
        path2 = linmdtw.mrmsdtw(X, Y, tau=10**4)[1]
        path2 = np.array(path2)
        path3 = linmdtw.mrmsdtw(X, Y, tau=10**2)[1]
        path3 = np.array(path3)
        err1 = linmdtw.get_alignment_row_col_dists(path1, path2)
        err2 = linmdtw.get_alignment_row_col_dists(path1, path3)
        assert(np.mean(err1) <= np.mean(err2))
        assert(linmdtw.get_path_cost(X, Y, path2) < linmdtw.get_path_cost(X, Y, path3))

    def test_mrmsdtw_refine(self):
        """
        测试在细化之后误差是否降低
        """
        X, Y = get_pcs(1000)
        path1 = linmdtw.dtw_brute_backtrace(X, Y)[1]
        path1 = np.array(path1)
        path2 = linmdtw.mrmsdtw(X, Y, tau=10**3, refine=True)[1]
        path2 = np.array(path2)
        path3 = linmdtw.mrmsdtw(X, Y, tau=10**3, refine=False)[1]
        path3 = np.array(path3)
        err1 = linmdtw.get_alignment_row_col_dists(path1, path2)
        err2 = linmdtw.get_alignment_row_col_dists(path1, path3)
        assert(np.mean(err1) <= np.mean(err2))
        assert(linmdtw.get_path_cost(X, Y, path2) < linmdtw.get_path_cost(X, Y, path3))