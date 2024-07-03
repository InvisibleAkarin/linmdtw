# distutils: language = c++

cimport linmdtwPy2Cpp
cimport cnp
import cython
import numpy

@cython.boundscheck(False)
@cython.wraparound(False)
def linmdtw(numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None, bint do_gpu, dict metadata):
    cost, path = linmdtw(X, Y, None, 500, do_gpu, metadata)
    return (cost, path)