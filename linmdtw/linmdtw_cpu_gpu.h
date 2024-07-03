// #ifndef LINMDTW_H
// #define LINMDTW_H
#ifndef LINMDTW_CPU_GPU_H
#define LINMDTW_CPU_GPU_H


#include <iostream>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>
#include <cuda_runtime.h>
#include <utility>
#include <limits>
using namespace std;

/** dynseqalign.pyx 中返回值的样子：
    ret = {'cost':cost, 'P':P}
    if debug == 1:
        ret['U'] = U
        ret['L'] = L
        ret['UL'] = UL
        ret['S'] = S
    return ret
 */
struct DTWBruteResult {
    float cost;
    vector<vector<int>> P;
    vector<vector<float>> U, L, UL;
    vector<vector<float>> S;
};

/** dtw.py 文件中记录的返回值结构：
    'cost': float
        对齐的最优代价（如果计算没有提前停止），
    'U'/'L'/'UL': ndarray(M, N)
        选择矩阵（如果调试），
    'd0'/'d1'/'d2': ndarray(min(M, N))
        如果选择了保存索引，则保存的行，
    'csm0'/'csm1'/'csm2': ndarray(min(M, N))
        如果选择了保存索引，则保存的交叉相似度距离
 */
struct DTWDiagResult {
    float cost;
    vector<float> d0, d1, d2;
    vector<float> csm0, csm1, csm2;
    vector<vector<float>> U, L, UL, S, CSM;
};

/**在dtw.py中的注释：
 *     返回
    -------
        (float: cost, ndarray(K, 2): 最优变形路径)
 */
struct DTWResult {
    float cost;
    std::vector<std::pair<int, int>> path;
};


void DTW_Diag_Step(vector<float>& d0, vector<float>& d1, vector<float>& d2, vector<float>& csm0, vector<float>& csm1, vector<float>& csm2, const vector<vector<float>>& X, const vector<vector<float>>& Y, int diagLen, const vector<int>& box, int reverse, int i, int debug, vector<vector<float>>& U, vector<vector<float>>& L, vector<vector<float>>& UL, vector<vector<float>>& S);
DTWResult dtw_brute_backtrace(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y, bool debug);
DTWBruteResult DTW(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y, int debug);
DTWResult linmdtw(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y, 
                  const std::vector<int>& box, int min_dim, bool do_gpu, const std::vector<int>& metadata);
DTWDiagResult dtw_diag(const vector<vector<float>>& X, const vector<vector<float>>& Y, int k_save, int k_stop, vector<int> box, bool reverse, bool debug, std::map<std::string, long double>* metadata);
DTWDiagResult wrap_dtw_diag_gpu(vector<vector<float>>& X, vector<vector<float>>& Y, int k_save, int k_stop, vector<int> box, bool reverse, bool debug, std::map<std::string, long double>* metadata);
void check_euclidean_inputs(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y);
int get_diag_len(const std::vector<int>& box, int k);
void update_min_cost(const std::vector<float>& d1, const std::vector<float>& d2, 
                     const std::vector<float>& csm, float& min_cost, std::vector<int>& min_idxs, 
                     int k, const std::vector<int>& box, const std::vector<std::vector<float>>& X, 
                     const std::vector<std::vector<float>>& Y);

#endif // LINMDTW_CPU_GPU_H

