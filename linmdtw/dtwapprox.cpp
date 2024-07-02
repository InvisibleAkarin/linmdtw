#include <math.h>
#include <float.h>

#define LEFT 0
#define UP 1
#define DIAG 2

void c_fastdtw_dynstep(float* X, float* Y, int d, int* I, int* J, int* left, int* up, int* diag, float* S, int* P, int N) {
    float dist;
    for (int idx = 0; idx < N; idx++) {
        // 第一步：计算欧几里得距离
        dist = 0.0;
        for (int k = 0; k < d; k++) {
            double diff = (X[I[idx]*d + k] - Y[J[idx]*d + k]);
            dist += diff*diff;
        }
        dist = sqrt(dist);

        // 第二步：执行动态规划步骤
        float score = -1;
        if (idx == 0) {
            score = 0;
        }
        else {
            // 左侧
            float leftScore = -1;
            if (left[idx] >= 0) {
                leftScore = S[left[idx]];
            }
            // 上方
            float upScore = -1;
            if (up[idx] >= 0) {
                upScore = S[up[idx]];
            }
            // 对角线
            float diagScore = -1;
            if (diag[idx] >= 0) {
                diagScore = S[diag[idx]];
            }

            if (leftScore > -1) {
                score = leftScore;
                P[idx] = LEFT;
            }
            if (upScore > -1 && (upScore < score || score == -1)) {
                score = upScore;
                P[idx] = UP;
            }
            if (diagScore > -1 && (diagScore <= score || score == -1)) {
                score = diagScore;
                P[idx] = DIAG;
            }
        }
        S[idx] = score + dist;

    }
}
