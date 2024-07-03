#include "dtw_diag_gpu.h"
#include "DTWGPU.h"
#include "alignmenttools.h"

// 初始化GPU的变量
bool DTW_GPU_Initialized = false;
bool DTW_GPU_Failed = false;
void init_gpu() {
    if (!DTW_GPU_Initialized) {
        try {
            // 初始化CUDA
            cudaFree(0); // 触发CUDA初始化
            DTW_GPU_Initialized = true;
            cout << "CUDA initialized successfully." << endl;
        }
        catch (const std::exception& e) {
            DTW_GPU_Failed = true;
            cerr << "Unable to initialize CUDA: " << e.what() << endl;
        }
    }
}
//// GPU版本的DTW对角线计算函数

map<string, vector<float>> dtw_diag_gpu(vector<vector<float>>& X, vector<vector<float>>& Y, int k_save, int k_stop, vector<int> box, bool reverse, bool debug,std::map<std::string, long double>* metadata) {
    assert(X[0].size() == Y[0].size());

    if (!DTW_GPU_Initialized) {
        init_gpu();
    }

    if (DTW_GPU_Failed) {
        cerr << "CUDA initialization failed. Exiting dtw_diag_gpu." << endl;
        return {};
    }

    if (!metadata) {
        metadata = new std::map<std::string, long double>();
    }

    int M = box.empty() ? X.size() : box[1] - box[0] + 1;
    int N = box.empty() ? Y.size() : box[3] - box[2] + 1;

    if (k_stop == -1) {
        k_stop = M + N - 2;
    }
    if (box.empty()) {
        box = {0, int(X.size()) - 1, 0, int(Y.size()) - 1};
    }

    int diagLen = min(M, N);
    int threadsPerBlock = min(diagLen, 512);
    int gridSize = static_cast<int>(ceil(diagLen / static_cast<float>(threadsPerBlock)));

    // 分配GPU内存
    float* d_X; float* d_Y; int* d_box;
    float *d0, *d1, *d2, *csm0, *csm1, *csm2;
    float *U, *L, *UL, *S;

    cudaMalloc(&d_X, X.size() * X[0].size() * sizeof(float));
    cudaMalloc(&d_Y, Y.size() * Y[0].size() * sizeof(float));
    cudaMalloc(&d_box, 4 * sizeof(int));

    cudaMalloc(&d0, diagLen * sizeof(float));
    cudaMalloc(&d1, diagLen * sizeof(float));
    cudaMalloc(&d2, diagLen * sizeof(float));
    cudaMalloc(&csm0, diagLen * sizeof(float));
    cudaMalloc(&csm1, diagLen * sizeof(float));
    cudaMalloc(&csm2, diagLen * sizeof(float));

    cudaMalloc(&U, debug ? M * N * sizeof(float) : sizeof(float));
    cudaMalloc(&L, debug ? M * N * sizeof(float) : sizeof(float));
    cudaMalloc(&UL, debug ? M * N * sizeof(float) : sizeof(float));
    cudaMalloc(&S, debug ? M * N * sizeof(float) : sizeof(float));

    // 将数据传输到GPU
    vector<float> flat_X(X.size() * X[0].size());
    vector<float> flat_Y(Y.size() * Y[0].size());
    for (size_t i = 0; i < X.size(); ++i) {
        copy(X[i].begin(), X[i].end(), flat_X.begin() + i * X[0].size());
    }
    for (size_t i = 0; i < Y.size(); ++i) {
        copy(Y[i].begin(), Y[i].end(), flat_Y.begin() + i * Y[0].size());
    }

    cudaMemcpy(d_X, flat_X.data(), flat_X.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, flat_Y.data(), flat_Y.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_box, box.data(), 4 * sizeof(int), cudaMemcpyHostToDevice);

    vector<float> d0_host(diagLen, 0.0f), d1_host(diagLen, 0.0f), d2_host(diagLen, 0.0f);
    vector<float> csm0_host(diagLen, 0.0f), csm1_host(diagLen, 0.0f), csm2_host(diagLen, 0.0f);

    cudaMemcpy(d0, d0_host.data(), diagLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d1, d1_host.data(), diagLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d2, d2_host.data(), diagLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(csm0, csm0_host.data(), diagLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(csm1, csm1_host.data(), diagLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(csm2, csm2_host.data(), diagLen * sizeof(float), cudaMemcpyHostToDevice);

    map<string, vector<float>> res;
    for (int k = 0; k < M+N-1; ++k) {
        DTW_Diag_Step<<<gridSize, threadsPerBlock>>>(d0, d1, d2, csm0, csm1, csm2, d_X, d_Y, X[0].size(), diagLen, d_box, reverse, k, debug, U, L, UL, S);
        cudaDeviceSynchronize();

        // 从GPU复制数据回主机内存
        cudaMemcpy(csm2_host.data(), csm2, csm2_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(csm0_host.data(), csm0, csm0_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(csm1_host.data(), csm1, csm1_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d0_host.data(), d0, diagLen * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d1_host.data(), d1, diagLen * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d2_host.data(), d2, diagLen * sizeof(float), cudaMemcpyDeviceToHost);

        update_alignment_metadata(*metadata, diagLen);

        if (k == k_save) {
            res["d0"] = vector<float>(d0_host.begin(), d0_host.end());
            res["csm0"] = vector<float>(csm0_host.begin(), csm0_host.begin() + diagLen);
            res["d1"] = vector<float>(d1_host.begin(), d1_host.end());
            res["csm1"] = vector<float>(csm1_host.begin(), csm1_host.begin() + diagLen);
            res["d2"] = vector<float>(d2_host.begin(), d2_host.end());
            res["csm2"] = vector<float>(csm2_host.begin(), csm2_host.begin() + diagLen);
        }
        if (k < k_stop) {
            swap(d0, d1);
            swap(d1, d2);
            swap(csm0, csm1);
            swap(csm1, csm2);

            swap(d0_host, d1_host);
            swap(d1_host, d2_host);
            swap(csm0_host, csm1_host);
            swap(csm1_host, csm2_host);
        }
    }

    vector<float> cost_d2(1), cost_csm2(1);
    cudaMemcpy(cost_d2.data(), d2, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cost_csm2.data(), csm2, sizeof(float), cudaMemcpyDeviceToHost);
    res["cost"] = vector<float>(1, cost_d2[0] + cost_csm2[0]);

    if (debug) {
        vector<float> U_host(M * N), L_host(M * N), UL_host(M * N), S_host(M * N);
        cudaMemcpy(U_host.data(), U, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(L_host.data(), L, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(UL_host.data(), UL, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(S_host.data(), S, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        res["U"] = U_host;
        res["L"] = L_host;
        res["UL"] = UL_host;
        res["S"] = S_host;
    }

    // 释放GPU内存
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_box);
    cudaFree(d0);
    cudaFree(d1);
    cudaFree(d2);
    cudaFree(csm0);
    cudaFree(csm1);
    cudaFree(csm2);
    cudaFree(U);
    cudaFree(L);
    cudaFree(UL);
    cudaFree(S);

    return res;
}
