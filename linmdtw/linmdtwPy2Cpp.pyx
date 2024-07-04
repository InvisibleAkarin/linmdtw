# distutils: language = c++

#cimport linmdtwPy2Cpp
import cython
# distutils: language = c++
from libcpp.vector cimport vector
import numpy as np
cimport numpy as cnp
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from cython.operator cimport dereference as deref, preincrement as inc


ctypedef vector[float] VectorFloat
ctypedef vector[VectorFloat] VectorVectorFloat
ctypedef map[string, double] StringToDoubleMap


cdef extern from "linmdtw.cpp":
    cdef cppclass DTWResult:
        float cost
        vector[pair[int, int]] path

    # 声明linmdtw函数，它返回一个DTWResult对象
    DTWResult linmdtw(VectorVectorFloat& X, VectorVectorFloat& Y, vector[int]& box, int min_dim, bool do_gpu, StringToDoubleMap* metadata)


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

cdef map[string, double] dict_to_map(dict py_dict):
    cdef:
        map[string, double] cpp_map = map[string, double]()
        string key
        double value
    for py_key, py_value in py_dict.items():
        key = py_key.encode('utf-8')
        value = py_value
        cpp_map[key] = value
    return cpp_map

cdef dict map_to_dict(const map[string, double]& cpp_map):
    cdef:
        dict py_dict = {}
        map[string, double].iterator it = cpp_map.begin()
        string key
        double value
    while it != cpp_map.end():
        key = deref(it).first.decode('utf-8')  # 使用 it.first 访问键，并解码为 UTF-8 字符串
        value = deref(it).second  # 使用 it.second 访问值
        py_dict[key] = value
        # 使用 ++it 来增加迭代器
        # 注意：这里的实现方式可能需要根据你的 Cython 环境进行调整
        inc(it)
    return py_dict



#@cython.boundscheck(False)
#@cython.wraparound(False)
def c_linmdtw(cnp.ndarray[float,ndim=2,mode="c"] X not None, cnp.ndarray[float,ndim=2,mode="c"] Y not None, bint do_gpu, dict metadata):
    cdef: 
        vector[vector[float]] X_vec = numpy_to_vector_vector_float(X)
        vector[vector[float]] Y_vec = numpy_to_vector_vector_float(Y)
        vector[int] box_vec = vector[int]()
        map[string, double] metadata_map = dict_to_map(metadata)
    cdef DTWResult result = linmdtw(X_vec, Y_vec, box_vec, 500, do_gpu, &metadata_map)
    metadata = map_to_dict(metadata_map)
    path_py = [(p.first, p.second) for p in result.path]
    cost = result.cost
    return (cost,path_py)