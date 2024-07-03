# distutils: language = c++

cimport linmdtwPy2Cpp
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def DTW(numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None, int debug):
    cost, path = linmdtw(X, Y, None, 500, True, None)
    return (cost, path)