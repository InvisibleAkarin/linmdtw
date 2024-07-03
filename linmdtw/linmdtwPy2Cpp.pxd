cdef extern from "linmdtw_cpu_gpu.cpp":
    float linmdtw(float* X, float* Y, int* P, int M, int N, int d, int debug, float* U, float* L, float* UL, float* S)
