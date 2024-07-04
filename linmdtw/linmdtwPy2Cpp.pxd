# linmdtwPy2Cpp.pxd
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "<vector>" namespace "std":
    pass  # 只是为了引入std::vector

cdef extern from "linmdtw.cpp":
    cdef cppclass DTWResult:
        float cost
        vector[pair[int, int]] path

    # 声明linmdtw函数，它返回一个DTWResult对象
    DTWResult linmdtw(vector<vector<float>>& X, vector<vector<float>>& Y, vector<int> box, int min_dim, bool do_gpu, std::map<std::string, long double>* metadata)

