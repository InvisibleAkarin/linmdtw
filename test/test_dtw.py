import pytest
import numpy as np
import linmdtw

class TestLibrary:
    # 库是否在作用域内安装？对象是否在作用域内？
    def test_import(self):
        import linmdtw
        from linmdtw import linmdtw, dtw_brute, dtw_brute_backtrace, fastdtw, mrmsdtw
        assert 1


class TestDTW:
    def test_input_type_warning(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        with pytest.warns(UserWarning, match="is not 32-bit, so creating 32-bit version") as w:
            linmdtw.linmdtw(X, X)
        with pytest.warns(UserWarning, match="is not 32-bit, so creating 32-bit version") as w:
            linmdtw.dtw_brute_backtrace(X, X)

    def test_dimension_warning(self):
        """
        测试当行数少于列数时，确保会抛出警告
        """
        np.random.seed(0)
        X = np.random.rand(3, 10)
        with pytest.warns(UserWarning, match="列数多于行数") as w:
            linmdtw.linmdtw(X, X)
        with pytest.warns(UserWarning, match="列数多于行数") as w:
            linmdtw.dtw_brute_backtrace(X, X)
    
    def test_shape_error(self):
        """
        测试如果两个时间序列不在相同的欧几里得维度中，确保会抛出错误
        """
        np.random.seed(0)
        X = np.random.rand(10, 3)
        Y = np.random.rand(20, 4)
        with pytest.raises(ValueError):
            linmdtw.linmdtw(X, Y)

    def test_brute_vs_diag(self):
        """
        确保使用教科书中的“光栅顺序”算法与对角线分治算法之间只有很小的差异
        """
        import time
        N = 2000
        t1 = np.linspace(0, 1, N)
        t2 = t1**2
        X = np.zeros((N, 2), dtype=np.float32)
        X[:, 0] = np.cos(2*np.pi*t1)
        X[:, 1] = np.sin(4*np.pi*t1)
        Y = np.zeros_like(X)
        Y[:, 0] = np.cos(2*np.pi*t2)
        Y[:, 1] = np.sin(4*np.pi*t2)
        cost1, path1 = linmdtw.linmdtw(X, Y)
        path1 = np.array(path1)
        cost2, path2 = linmdtw.dtw_brute_backtrace(X, Y)
        assert(np.allclose(cost1, cost2))
        path2 = np.array(path2)
        err = linmdtw.get_alignment_row_col_dists(path1, path2)
        assert(np.mean(err) < 1)
        metadata = {'totalCells':0, 'M':X.shape[0], 'N':Y.shape[0], 'timeStart':time.time()}
        _, path3 = linmdtw.linmdtw(X, Y, do_gpu=False, metadata=metadata)
        path3 = np.array(path3)
        err = linmdtw.get_alignment_row_col_dists(path2, path3)
        assert(np.mean(err) < 1)
    
    def test_synthetic_warppath(self):
        """
        创建一个随机的合成扭曲路径，并验证 linmdtw 生成的扭曲路径在该路径的小误差范围内
        """
        np.random.seed(1)
        M = 2000
        N = 3000
        t1 = np.linspace(0, 1, N)
        X = np.zeros((N, 2), dtype=np.float32)
        X[:, 0] = np.cos(2*np.pi*t1)
        X[:, 1] = np.sin(4*np.pi*t1)
        D = linmdtw.alignmenttools.get_parameterization_dict(M)
        t2 = linmdtw.alignmenttools.sample_parameterization_dict(D, 2)
        Y = linmdtw.alignmenttools.get_interpolated_euclidean_timeseries(X, t2)

        path = linmdtw.linmdtw(X, Y)[1]
        pathorig = linmdtw.alignmenttools.param_to_warppath(t2, N)
        pathorig = np.array(pathorig, dtype=int)

        err_area = linmdtw.get_alignment_area_dist(path, pathorig)
        err_rowcol = np.mean(linmdtw.get_alignment_row_col_dists(path, pathorig))
        assert(err_area < 1)
        assert(err_rowcol < 1)