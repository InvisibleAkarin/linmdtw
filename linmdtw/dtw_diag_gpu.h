#ifndef DTW_DIAG_GPU_H
#define DTW_DIAG_GPU_H
// 初始化CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include <map>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <cmath>

using namespace std;
using namespace std::chrono;


void init_gpu();
map<string, vector<float>> dtw_diag_gpu(vector<vector<float>>& X, vector<vector<float>>& Y, int k_save = -1, int k_stop = -1, vector<int> box = {}, bool reverse = false, bool debug = false, std::map<std::string, long double>* metadata = nullptr);
#endif // DTW_DIAG_GPU_H
