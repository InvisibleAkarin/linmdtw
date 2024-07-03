# linmdtwPy2Cpp.pxd
from libcpp.vector cimport vector

cdef extern from "<vector>" namespace "std":
    pass  # 只是为了引入std::vector

cdef extern from "linmdtw.hpp":
cdef cppclass DTWResult:
        # 声明DTWResult类的公共接口
        float cost
        vector[float] d0, d1, d2
        vector[float] csm0, csm1, csm2
        vector[vector[float]] U, L, UL, S, CSM

    # 声明linmdtw函数，它返回一个DTWResult对象
    DTWResult linmdtw(const vector[vector[float]]& X, const vector[vector[float]]& Y, const vector[int]& box, int min_dim, bool do_gpu, const vector[int]& metadata)