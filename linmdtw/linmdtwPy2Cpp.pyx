# distutils: language = c++

cimport linmdtwPy2Cpp
cimport cnp
import cython
# distutils: language = c++
from libcpp.vector cimport vector
import numpy as np
cimport numpy as cnp
from libcpp.map cimport map
from libcpp.string cimport string



cdef extern from "linmdtw.cpp":
    cdef cppclass DTWResult:
        float cost
        vector[pair[int, int]] path

    # 声明linmdtw函数，它返回一个DTWResult对象
    DTWResult linmdtw(const vector[vector[float]]& X, const vector[vector[float]]& Y, const vector[int]& box, int min_dim, bool do_gpu, const vector[int]& metadata)


def convert_dtw_result_to_py(DTWResult result):
    # 将 C++ vector 转换为 Python 列表
    path_py = [(p.first, p.second) for p in result.path]
    # 创建并返回一个包含结果的元组
    return (result.cost, path_py)

cdef vector[vector[float]] numpy_to_vector_vector_float(cnp.ndarray[cnp.float32_t, ndim=2] arr):
    cdef:
        vector[vector[float]] vec = vector[vector[float]]()
        vector[float] inner_vec
        int i, j
    for i in range(arr.shape[0]):
        inner_vec = vector[float]()
        for j in range(arr.shape[1]):
            inner_vec.push_back(arr[i, j])
        vec.push_back(inner_vec)
    return vec

cdef vector[int] numpy_to_vector_int(cnp.ndarray[cnp.int32_t, ndim=1] arr):
    cdef:
        vector[int] vec = vector[int]()
        int i
    for i in range(arr.shape[0]):
        vec.push_back(arr[i])
    return vec

cdef map[string, long double] dict_to_map(dict py_dict):
    cdef:
        map[string, long double] cpp_map = map[string, long double]()
        string key
        long double value
    for py_key, py_value in py_dict.items():
        key = py_key.encode('utf-8')
        value = py_value
        cpp_map[key] = value
    return cpp_map

cdef dict map_to_dict(const map[string, long double]& cpp_map):
    cdef:
        dict py_dict = {}
        string key
        long double value
    for key, value in cpp_map.items():
        py_dict[key.decode('utf-8')] = value
    return py_dict

@cython.boundscheck(False)
@cython.wraparound(False)
def c_linmdtw(numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None, bint do_gpu, dict metadata):
    vector[vector[float]] X_vec = numpy_to_vector_vector_float(X)
    vector[vector[float]] Y_vec = numpy_to_vector_vector_float(Y)
    vector[int] box_vec = vector[int]()
    map[string, long double] metadata_map = dict_to_map(metadata)
    cdef DTWResult result = linmdtw(X_vec, Y_vec, box_vec, 500, do_gpu, &metadata_map)
    return convert_dtw_result_to_py(result)