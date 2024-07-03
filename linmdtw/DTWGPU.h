#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#ifndef DTWC_DTWGPU_H
#define DTWC_DTWGPU_H
__global__ void DTW_Diag_Step(float* d0, float* d1, float* d2, float* csm0, float* csm1, float* csm2, const float* X, const float* Y, int dim, int diagLen, int* box, int reverse, int i, int debug, float* U, float* L, float* UL, float* S);
#endif //DTWC_DTWGPU_H
