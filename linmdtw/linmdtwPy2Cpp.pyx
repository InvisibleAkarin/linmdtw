# distutils: language = c++

cimport linmdtwPy2Cpp
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def DTW(numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None, bool do_gpu, dict metadata):
    cost, path = linmdtw(X, Y, None, 500, do_gpu, metadata)
    return (cost, path)