#include<vector>

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
 * 看起来有点冗余，可以化简
 * 它用 DTW 函数
 * 它被 dtw_brute_backtrace 函数调用
 */
DTWBruteResult dtw_brute(const vector<vector<float>>& X, const vector<vector<float>>& Y, bool debug) {
    return DTW(X, Y, debug);
}

/**
 * 对应原先代码 dtw.py 中同名函数
 */
DTWResult dtw_brute_backtrace(const vector<vector<float>>& X, const vector<vector<float>>& Y, bool debug) {
    DTWBruteResult res = dtw_brute(X, Y, debug);

    int i = X.size() - 1;
    int j = Y.size() - 1;
    vector<pair<int, int>> path = {{i, j}};
    vector<vector<int>> step = {{0, -1}, {-1, 0}, {-1, -1}}; // LEFT, UP, DIAG

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

    vector<float> X_flat(M * d);
    vector<float> Y_flat(N * d);

    // Flatten X and Y matrices
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < d; ++j) {
            X_flat[i * d + j] = X[i][j];
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            Y_flat[i * d + j] = Y[i][j];
        }
    }

    vector<int> P_flat(M * N);
    vector<float> U_flat(M * N);
    vector<float> L_flat(M * N);
    vector<float> UL_flat(M * N);
    vector<float> S_flat(M * N);

    float cost = c_dtw(X_flat.data(), Y_flat.data(), P_flat.data(), M, N, d, debug, U_flat.data(), L_flat.data(), UL_flat.data(), S_flat.data());

    // Unflatten matrices
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
 */
void DTW_Diag_Step(vector<float>& d0, vector<float>& d1, vector<float>& d2, vector<float>& csm0, vector<float>& csm1, vector<float>& csm2, const vector<vector<float>>& X, const vector<vector<float>>& Y, int diagLen, const vector<int>& box, int reverse, int i, int debug, vector<vector<float>>& U, vector<vector<float>>& L, vector<vector<float>>& UL, vector<vector<float>>& S) {
    int dim = X[0].size();
    c_diag_step(d0.data(), d1.data(), d2.data(), csm0.data(), csm1.data(), csm2.data(), const_cast<float*>(X[0].data()), const_cast<float*>(Y[0].data()), dim, diagLen, const_cast<int*>(box.data()), reverse, i, debug, U[0].data(), L[0].data(), UL[0].data(), S[0].data());
}


/**
 * 对应 dtw.py中的同名函数
 * 它调用 DTW_Diag_Step函数
 * 它被 linmdtw 函数所调用
 */
DTWDiagResult dtw_diag(const vector<vector<float>>& X, const vector<vector<float>>& Y, int k_save, int k_stop, vector<int> box, bool reverse, bool debug, std::map<std::string, long double>* metadata) {
    if (box.empty()) {
        box = {0, static_cast<int>(X.size()) - 1, 0, static_cast<int>(Y.size()) - 1};
    }

    int M = box[1] - box[0] + 1;
    int N = box[3] - box[2] + 1;
    box = {box[0], box[1], box[2], box[3]}; // Ensure box is in the correct format
    if (k_stop == -1) {
        k_stop = M + N - 2;
    }
    if (k_save == -1) {
        k_save = k_stop;
    }

    // Debugging info
    vector<vector<float>> U(1, vector<float>(1, 0.0f)), L(1, vector<float>(1, 0.0f)), UL(1, vector<float>(1, 0.0f)), S(1, vector<float>(1, 0.0f)), CSM(1, vector<float>(1, 0.0f));
    if (debug) {
        U = vector<vector<float>>(M, vector<float>(N, 0.0f));
        L = vector<vector<float>>(M, vector<float>(N, 0.0f));
        UL = vector<vector<float>>(M, vector<float>(N, 0.0f));
        S = vector<vector<float>>(M, vector<float>(N, 0.0f));
        CSM = vector<vector<float>>(M, vector<float>(N, 0.0f));
    }

    // Diagonals
    int diagLen = min(M, N);
    vector<float> d0(diagLen, 0.0f), d1(diagLen, 0.0f), d2(diagLen, 0.0f);
    vector<float> csm0(diagLen, 0.0f), csm1(diagLen, 0.0f), csm2(diagLen, 0.0f);
    int csm0len = diagLen, csm1len = diagLen, csm2len = diagLen;

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
        if (metadata) {
            update_alignment_metadata(*metadata, csm2len);
        }
        if (k == k_save) {
            res.d0 = d0;
            res.csm0 = csm0;
            res.d1 = d1;
            res.csm1 = csm1;
            res.d2 = d2;
            res.csm2 = csm2;
        }
        if (k < k_stop) {
            // Shift diagonals (triple buffering)
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
// dtw_diag_gpu




/**************************************************************
 *    cpu与gpu计算的公共部分
 *    对应原先代码中的 dtw.py
 ************************************************************/

/**
 * 对应原先代码中 dtw.py 中的同名函数
 * 被 linmdwt 函数所调用
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
    check_euclidean_inputs(X, Y);
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

    if (M < min_dim || N < min_dim) {
        if (metadata) {
            (*metadata)["totalCells"] += M * N;
        }
        // 提取 X 和 Y 的子矩阵
        vector<vector<float>> sub_X(M, vector<float>(X[0].size()));
        vector<vector<float>> sub_Y(N, vector<float>(Y[0].size()));

        for (int i = 0; i < M; ++i) {
            sub_X[i] = X[box[0] + i];
        }

        for (int j = 0; j < N; ++j) {
            sub_Y[j] = Y[box[2] + j];
        }

        auto result = dtw_brute_backtrace(sub_X, sub_Y);
        for (auto& p : result.path) {
            p.first += box[0];
            p.second += box[2];
        }
        return result;
    }

    int K = M + N - 1;
    int k_save = static_cast<int>(ceil(K / 2.0));
    auto res1 = dtw_diag_fn(X, Y, k_save, k_save, box, false, false, metadata);

    int k_save_rev = k_save;
    if (K % 2 == 0) {
        k_save_rev += 1;
    }
    auto res2 = dtw_diag_fn(X, Y, k_save_rev, k_save_rev, box, true, false, metadata);
    // Swap d0 and d2, csm0 and csm2 in res2
    swap(res2.d0, res2.d2);
    swap(res2.csm0, res2.csm2);

    //Chop off extra diagonal elements
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

    reverse(res2.d0.begin(), res2.d0.end());
    reverse(res2.d1.begin(), res2.d1.end());
    reverse(res2.d2.begin(), res2.d2.end());
    reverse(res2.csm0.begin(),res2.csm0.end());
    reverse(res2.csm1.begin(),res2.csm1.end());
    reverse(res2.csm2.begin(),res2.csm2.end());

    float min_cost = numeric_limits<float>::infinity();
    vector<int> min_idxs(2);

    update_min_cost(res1.d0, res2.d0, res2.csm0, min_cost, min_idxs, k_save - 2 + 0, box, X, Y);
    update_min_cost(res1.d1, res2.d1, res2.csm1, min_cost, min_idxs, k_save - 2 + 1, box, X, Y);
    update_min_cost(res1.d2, res2.d2, res2.csm2, min_cost, min_idxs, k_save - 2 + 2, box, X, Y);
    //分治
    vector<pair<int, int>> left_path;
    vector<int> box_left = {box[0], min_idxs[0], box[2], min_idxs[1]};
    left_path = linmdtw(X, Y, box_left, min_dim, do_gpu, metadata).path;

    vector<pair<int, int>> right_path;
    vector<int> box_right = {min_idxs[0], box[1], min_idxs[1], box[3]};

    right_path = linmdtw(X, Y, box_right, min_dim, do_gpu, metadata).path;
//    std::cout << min_idxs[0] << min_idxs[1] <<box[0] <<box[1] << box[2] <<box[3]<<endl;
//    for (const auto& p : right_path) {
//        std::cout << "(" << p.first << ", " << p.second << ") ";
//    }
    left_path.insert(left_path.end(), right_path.begin() + 1, right_path.end());

    return {min_cost, left_path};
}