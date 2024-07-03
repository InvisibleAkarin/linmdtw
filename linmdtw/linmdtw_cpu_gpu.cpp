#include "linmdtw_cpu_gpu.h"

/*************************************************************
 *    cpu计算linmdtw部分
 *    对应原先代码中的 dynseqalign.pyx 和 一部分
 *    调用 dtw.cpp
*************************************************************/

/**
 * 这个函数不是直接转化来的。别的组的参考代码中有这个。先注释上。
 */
//将二维vector转换为一维array
// template <typename T>
// T* to_1d_array(vector<vector<T>>& matrix) {
//     T* array = new T[matrix.size() * matrix[0].size()];
//     for (size_t i = 0; i < matrix.size(); ++i) {
//         for (size_t j = 0; j < matrix[i].size(); ++j) {
//             array[i * matrix[0].size() + j] = matrix[i][j];
//         }
//     }
//     return array;
// }

/**
 * 对应原来代码中 dtw.py 中同名函数
 * 它用 DTW 函数 使用bf方法计算dtw
 * 它被 dtw_brute_backtrace 函数调用
 * 冗余
 */
// DTWBruteResult dtw_brute(const vector<vector<float>>& X, const vector<vector<float>>& Y, bool debug) {
//     return DTW(X, Y, debug);
// }

/**
 * 对应原先代码 dtw.py 中同名函数
 * 调用DWT（修改后）：使用bf方法计算dtw
 * 被linmdtw所调用
 */
DTWResult dtw_brute_backtrace(const vector<vector<float>>& X, const vector<vector<float>>& Y, bool debug) {
    // 先执行bf法计算dtw dp
    DTWBruteResult res = DTW(X, Y, debug);

    int i = X.size() - 1;
    int j = Y.size() - 1;
    vector<pair<int, int>> path = {{i, j}};
    vector<vector<int>> step = {{0, -1}, {-1, 0}, {-1, -1}}; // 左，上，对角线
    while (!(path.back().first == 0 && path.back().second == 0)) {
        auto s = step[res.P[i][j]];
        i += s[0];
        j += s[1];
        path.emplace_back(i, j);
    }
    reverse(path.begin(), path.end());
    DTWResult result;
    result.cost = res.cost;
    result.path = path;
    return result;
}

/**
 * 对应 dynseqalign.pyx中的同名函数
 * 现在这一部分不使用cython了，可以化简
 * 它调用 dwt.cpp 中的cpu计算代码 c_dtw
 * 它被 dtw_brute 调用
 */
DTWBruteResult DTW(const vector<vector<float>>& X, const vector<vector<float>>& Y, int debug) {
    int M = X.size();
    int N = Y.size();
    int d = Y[0].size();
    vector<vector<int>> P(M, vector<int>(N, 0));
    vector<vector<float>> U, L, UL;
    if (debug == 1) {
        U.resize(M, vector<float>(N, 0.0));
        L.resize(M, vector<float>(N, 0.0));
        UL.resize(M, vector<float>(N, 0.0));
    } else {
        U.resize(1, vector<float>(1, 0.0));
        L.resize(1, vector<float>(1, 0.0));
        UL.resize(1, vector<float>(1, 0.0));
    }
    vector<vector<float>> S(M, vector<float>(N, 0.0));
    // 接下来准备调用 dtw.cpp中的c_dtw函数
    vector<float> X1d(M * d);
    vector<float> Y1d(N * d);
    // 
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < d; ++j) {
            X1d[i * d + j] = X[i][j];
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            Y1d[i * d + j] = Y[i][j];
        }
    }
    vector<int> P1d(M * N);
    vector<float> U1d(M * N);
    vector<float> L1d(M * N);
    vector<float> UL1d(M * N);
    vector<float> S1d(M * N);
    
    // 调用dtw.cpp中cpu计算代码
    float cost = c_dtw(X1d.data(), Y1d.data(), P1d.data(), M, N, d, debug, U1d.data(), L1d.data(), UL1d.data(), S1d.data());

    // 再将 1d 的数组重新整理回 2d
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            P[i][j] = P_flat[i * N + j];
            S[i][j] = S_flat[i * N + j];
            if (debug == 1) {
                U[i][j] = U_flat[i * N + j];
                L[i][j] = L_flat[i * N + j];
                UL[i][j] = UL_flat[i * N + j];
            }
        }
    }

    // 填充返回值 DTWBruteResult 的各个字段
/** 参考 dynseqalign.pyx 中返回值的样子：
    ret = {'cost':cost, 'P':P}
    if debug == 1:
        ret['U'] = U
        ret['L'] = L
        ret['UL'] = UL
        ret['S'] = S
    return ret
 */
    DTWBruteResult ret;
    ret.cost = cost;
    ret.P = P;
    if (debug == 1) {
        ret.U = U;
        ret.L = L;
        ret.UL = UL;
    }
    ret.S = S;
    return ret;
}


/**
 * 对应 dynseqalign.pyx中的同名函数
 * 原先是cython代码，现在没必要在此使用cython了，可以化简
 * 它调用 dtw.cpp 中的 c_diag_step
 * 它被 dtw_diag 调用
 * 
 * 之前的cython代码：
    def DTW_Diag_Step(numpy.ndarray[float,ndim=1,mode="c"] d0 not None, numpy.ndarray[float,ndim=1,mode="c"] d1 not None, numpy.ndarray[float,ndim=1,mode="c"] d2 not None, numpy.ndarray[float,ndim=1,mode="c"] csm0 not None, numpy.ndarray[float,ndim=1,mode="c"] csm1 not None, numpy.ndarray[float,ndim=1,mode="c"] csm2 not None, numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None, int diagLen, numpy.ndarray[int,ndim=1,mode="c"] box not None, int reverse, int i, int debug, numpy.ndarray[float,ndim=2,mode="c"] U not None, numpy.ndarray[float,ndim=2,mode="c"] L not None, numpy.ndarray[float,ndim=2,mode="c"] UL not None, numpy.ndarray[float,ndim=2,mode="c"] S not None):
    cdef int dim = X.shape[1]
    dynseqalign.c_diag_step(&d0[0], &d1[0], &d2[0], &csm0[0], &csm1[0], &csm2[0], &X[0, 0], &Y[0, 0], dim, diagLen, &box[0], reverse, i, debug, &U[0, 0], &L[0, 0], &UL[0, 0], &S[0, 0])
 */

void DTW_Diag_Step(vector<float>& d0, vector<float>& d1, vector<float>& d2, vector<float>& csm0, vector<float>& csm1, vector<float>& csm2, const vector<vector<float>>& X, const vector<vector<float>>& Y, int diagLen, const vector<int>& box, int reverse, int i, int debug, vector<vector<float>>& U, vector<vector<float>>& L, vector<vector<float>>& UL, vector<vector<float>>& S) {
    int dim = X[0].size();
    c_diag_step(d0.data(), d1.data(), d2.data(), csm0.data(), csm1.data(), csm2.data(), const_cast<float*>(X[0].data()), const_cast<float*>(Y[0].data()), dim, diagLen, const_cast<int*>(box.data()), reverse, i, debug, U[0].data(), L[0].data(), UL[0].data(), S[0].data());
}

// 包装函数，用于统一 dtw_diag 和 dtw_diag_gpu 的返回类型
DTWDiagResult wrap_dtw_diag_gpu(vector<vector<float>>& X, vector<vector<float>>& Y, int k_save, int k_stop, vector<int> box, bool reverse, bool debug, std::map<std::string, long double>* metadata) {
    map<string, vector<float>> gpu_result = dtw_diag_gpu(X, Y, k_save, k_stop, box, reverse, debug, metadata);
    DTWDiagResult result;
    result.cost = gpu_result["cost"][0];
    if (debug) {
        result.U = {gpu_result["U"]};
        result.L = {gpu_result["L"]};
        result.UL = {gpu_result["UL"]};
        result.S = {gpu_result["S"]};
    }
    result.d0 = gpu_result["d0"];
    result.d1 = gpu_result["d1"];
    result.d2 = gpu_result["d2"];
    result.csm0 = gpu_result["csm0"];
    result.csm1 = gpu_result["csm1"];
    result.csm2 = gpu_result["csm2"];
    return result;
}

/**
 * 对应 dtw.py中的同名函数
 * 它调用 DTW_Diag_Step函数
 * 它被 linmdtw 函数所调用
 * 
 * py中的注释：    
    返回
    -------
    {
        'cost': float
            对齐的最优代价（如果计算没有提前停止），
        'U'/'L'/'UL': ndarray(M, N)
            选择矩阵（如果调试），
        'd0'/'d1'/'d2': ndarray(min(M, N))
            如果选择了保存索引，则保存的行，
        'csm0'/'csm1'/'csm2': ndarray(min(M, N))
            如果选择了保存索引，则保存的交叉相似度距离
    }
 */
DTWDiagResult dtw_diag(const vector<vector<float>>& X, const vector<vector<float>>& Y, int k_save, int k_stop, vector<int> box, bool reverse, bool debug, std::map<std::string, long double>* metadata) {
    //box: 一个 [startx, endx, starty, endy] 的列表
    if (box.empty()) {
        box = {0, static_cast<int>(X.size()) - 1, 0, static_cast<int>(Y.size()) - 1};
    }
    // 
    int M = box[1] - box[0] + 1;
    int N = box[3] - box[2] + 1;
    box = {box[0], box[1], box[2], box[3]};
    if (k_stop == -1) {  // 停止计算的对角线 d2 的索引
        k_stop = M + N - 2;
    }
    if (k_save == -1) {  // 保存 d0、d1 和 d2 的对角线 d2 的索引
        k_save = k_stop;
    }

    // 调试信息
    vector<vector<float>> U(1, vector<float>(1, 0.0f));
    vector<vector<float>> L(1, vector<float>(1, 0.0f));
    vector<vector<float>> UL(1, vector<float>(1, 0.0f));
    vector<vector<float>> S(1, vector<float>(1, 0.0f));
    vector<vector<float>> CSM(1, vector<float>(1, 0.0f));
    if (debug) {
        U = vector<vector<float>>(M, vector<float>(N, 0.0f));
        L = vector<vector<float>>(M, vector<float>(N, 0.0f));
        UL = vector<vector<float>>(M, vector<float>(N, 0.0f));
        S = vector<vector<float>>(M, vector<float>(N, 0.0f));
        CSM = vector<vector<float>>(M, vector<float>(N, 0.0f));
    }

    // 对角线
    int diagLen = min(M, N);
    vector<float> d0(diagLen, 0.0f);
    vector<float> d1(diagLen, 0.0f);
    vector<float> d2(diagLen, 0.0f);

    // 沿对角线的点之间的距离
    vector<float> csm0(diagLen, 0.0f);
    vector<float> csm1(diagLen, 0.0f);
    vector<float> csm2(diagLen, 0.0f);
    int csm0len = diagLen;
    int csm1len = diagLen;
    int csm2len = diagLen;

    // 遍历对角线
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
    DTWDiagResult res;
    for (int k = 0; k <= k_stop; ++k) {
        DTW_Diag_Step(d0, d1, d2, csm0, csm1, csm2,X, Y,
                      diagLen, box, reverse, k, debug,
                      U, L, UL, S);
        csm2len = get_diag_len(box, k);
        if (debug) {
            auto [i, j] = get_diag_indices(M, N, k);
            for (size_t idx = 0; idx < i.size(); ++idx) {
                CSM[i[idx]][j[idx]] = csm2[idx];
            }
        }
        if (metadata) {  // 用于存储计算信息的字典
            update_alignment_metadata(*metadata, csm2len);
        }
        if (k == k_save) {  // 这里有没有copy的问题？
            res.d0 = d0;
            res.csm0 = csm0;
            res.d1 = d1;
            res.csm1 = csm1;
            res.d2 = d2;
            res.csm2 = csm2;
        }
        if (k < k_stop) {
            // 移动对角线（三重缓冲）
            swap(d0, d1);
            swap(d1, d2);
            swap(csm0, csm1);
            swap(csm1, csm2);
            swap(csm0len, csm1len);
            swap(csm1len, csm2len);
        }
    }
    res.cost = d2[0] + csm2[0];
    if (debug) {
        res.U = U;
        res.L = L;
        res.UL = UL;
        res.S = S;
        res.CSM = CSM;
    }
    return res;
}

/*************************************************************
*       gpu计算linmdtw部分
*     对应原先代码中的 dtwgpu.py
*************************************************************/
// init_gpu
// dtw_diag_gpu



/**************************************************************
 *    cpu与gpu计算的公共部分
 *    对应原先代码中的 dtw.py
 ************************************************************/

/**
 * 对应原先代码中 dtw.py 中的同名函数
 * 被 linmdwt 函数所调用
 * 
 * ！！与原函数不等价
 */
void check_euclidean_inputs(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y) {
    if (X[0].size() != Y[0].size()) {
        throw std::invalid_argument("The input time series are not in the same dimension space");
    }
    if (X.size() < X[0].size()) {
        std::cerr << "Warning: X has more columns than rows; did you mean to transpose?" << std::endl;
    }
    if (Y.size() < Y[0].size()) {
        std::cerr << "Warning: Y has more columns than rows; did you mean to transpose?" << std::endl;
    }
}

/**
 * 原先代码中似乎没有
 * 被 linmdtw 调用
 */
void update_min_cost(const vector<float>& dleft, const vector<float>& dright, const vector<float>& csmright, float& min_cost, vector<int>& min_idxs, int k, const vector<int>& box, const vector<vector<float>>& X, const vector<vector<float>>& Y) {
    vector<float> diagsum(dleft.size());
    for (size_t i = 0; i < diagsum.size(); ++i) {
        diagsum[i] = dleft[i] + dright[i] + csmright[i];
    }
    int idx = min_element(diagsum.begin(), diagsum.end()) - diagsum.begin();
    if (diagsum[idx] < min_cost) {
        min_cost = diagsum[idx];
        auto indices = get_diag_indices(X.size(), Y.size(), k, box);
        min_idxs[0] = indices.first[idx];
        min_idxs[1] = indices.second[idx];
    }
}

/**
 * 通过cython接口，为orchestral所调用，负责cpu和gpu上计算的协调
 */
DTWResult linmdtw(vector<vector<float>>& X, vector<vector<float>>& Y, vector<int> box, int min_dim, bool do_gpu, std::map<std::string, long double>* metadata) {
    // 计算 XY欧氏距离 = 对应点间差的平方求和开根号
    check_euclidean_inputs(X, Y);
    // 选择linmdtw计算使用cpu还是gpu
    function<DTWDiagResult(vector<vector<float>>&, vector<vector<float>>&, int, int, vector<int>, bool, bool, std::map<std::string, long double>*)> dtw_diag_fn = dtw_diag;
    if (do_gpu) {
        if (!DTW_GPU_Initialized) {
            init_gpu();
        }
        if (DTW_GPU_Failed) {
            cerr << "Falling back to CPU" << endl;
            do_gpu = false;
        } else {
            dtw_diag_fn = wrap_dtw_diag_gpu;
        }
    }

    if (box.empty()) {
        box = {0, static_cast<int>(X.size()) - 1, 0, static_cast<int>(Y.size()) - 1};
    }
    int M = box[1] - box[0] + 1;
    int N = box[3] - box[2] + 1;

    // 停止条件，回退到CPU
    if (M < min_dim || N < min_dim) {
        if (metadata) {
            (*metadata)["totalCells"] += M * N;
        }
        // 准备参数 X[box[0]:box[1]+1, :], Y[box[2]:box[3]+1, :]
        vector<vector<float>> sub_X(M, vector<float>(X[0].size()));
        vector<vector<float>> sub_Y(N, vector<float>(Y[0].size()));
        for (int i = 0; i < M; ++i) {
            sub_X[i] = X[box[0] + i];
        }
        for (int j = 0; j < N; ++j) {
            sub_Y[j] = Y[box[2] + j];
        }

        auto result = dtw_brute_backtrace(sub_X, sub_Y);
        // DTWResult result = dtw_brute_backtrace(sub_X, sub_Y);
        for (auto& p : result.path) {
            p.first += box[0];
            p.second += box[2];
        }
        return result;
    }

    // 否则，继续递归
    int K = M + N - 1;
    // 进行前向计算
    int k_save = static_cast<int>(ceil(K / 2.0));
    // 参数表不同了
    auto res1 = dtw_diag_fn(X, Y, k_save, k_save, box, false, false, metadata);

    // 进行后向计算
    int k_save_rev = k_save;
    if (K % 2 == 0) {
        k_save_rev += 1;
    }
    // 参数表同样不同了
    auto res2 = dtw_diag_fn(X, Y, k_save_rev, k_save_rev, box, true, false, metadata);
    // Swap d0 and d2, csm0 and csm2 in res2
    swap(res2.d0, res2.d2);
    swap(res2.csm0, res2.csm2);

    // 去掉多余的对角元素
    int sz = get_diag_len(box, k_save - 2);
    res1.d0.resize(sz);
    res1.csm0.resize(sz);
    sz = get_diag_len(box, k_save - 2+1);
    res1.d1.resize(sz);
    res1.csm1.resize(sz);
    sz = get_diag_len(box, k_save - 2+2);
    res1.d2.resize(sz);
    res1.csm2.resize(sz);

    sz = get_diag_len(box, k_save_rev - 2);
    res2.d0.resize(sz);
    res2.csm0.resize(sz);
    sz = get_diag_len(box, k_save - 2+1);
    res2.d1.resize(sz);
    res2.csm1.resize(sz);
    sz = get_diag_len(box, k_save - 2+2);
    res2.d2.resize(sz);
    res2.csm2.resize(sz);

    // 对齐反向对角线
    reverse(res2.d0.begin(), res2.d0.end());
    reverse(res2.d1.begin(), res2.d1.end());
    reverse(res2.d2.begin(), res2.d2.end());
    reverse(res2.csm0.begin(),res2.csm0.end());
    reverse(res2.csm1.begin(),res2.csm1.end());
    reverse(res2.csm2.begin(),res2.csm2.end());

    // 在三个对角线中找到最小成本并在该元素上拆分
    float min_cost = numeric_limits<float>::infinity();
    vector<int> min_idxs(2);

    update_min_cost(res1.d0, res2.d0, res2.csm0, min_cost, min_idxs, k_save - 2 + 0, box, X, Y);
    update_min_cost(res1.d1, res2.d1, res2.csm1, min_cost, min_idxs, k_save - 2 + 1, box, X, Y);
    update_min_cost(res1.d2, res2.d2, res2.csm2, min_cost, min_idxs, k_save - 2 + 2, box, X, Y);

    // 递归计算左路径
    vector<pair<int, int>> left_path;
    vector<int> box_left = {box[0], min_idxs[0], box[2], min_idxs[1]};
    left_path = linmdtw(X, Y, box_left, min_dim, do_gpu, metadata).path;

    // 递归计算右路径
    vector<pair<int, int>> right_path;
    vector<int> box_right = {min_idxs[0], box[1], min_idxs[1], box[3]};
    right_path = linmdtw(X, Y, box_right, min_dim, do_gpu, metadata).path;

    left_path.insert(left_path.end(), right_path.begin() + 1, right_path.end());
    return {min_cost, left_path};
}