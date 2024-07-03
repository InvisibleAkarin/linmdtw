# distutils: language = c++

cimport linmdtwPy2Cpp
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def DTW(numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None, int debug):
    c_linmdtw_cpu_gpu()
    return ...