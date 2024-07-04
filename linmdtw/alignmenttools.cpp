#include "alignmenttools.h"
// 获取对角线长度
int get_diag_len(const vector<int>& box, int k) {
    int M = box[1] - box[0] + 1;
    int N = box[3] - box[2] + 1;
    int starti = k;
    int startj = 0;

    if (k >= M) {
        starti = M - 1;
        startj = k - (M - 1);
    }
    int endj = k;
    int endi = 0;
    if (k >= N) {
        endj = N - 1;
        endi = k - (N - 1);
    }
    return endj - startj + 1;
}

// 获取对角线索引
pair<vector<int>, vector<int>> get_diag_indices(int MTotal, int NTotal, int k, const vector<int>& box, bool reverse) {
    vector<int> adjusted_box = box;
    if (adjusted_box.empty()) {
        adjusted_box = {0, MTotal - 1, 0, NTotal - 1};
    }

    int M = adjusted_box[1] - adjusted_box[0] + 1;
    int N = adjusted_box[3] - adjusted_box[2] + 1;
    int starti = k;
    int startj = 0;

    if (k > M - 1) {
        starti = M - 1;
        startj = k - (M - 1);
    }

    vector<int> i(starti + 1);
    vector<int> j(starti + 1);

    for (int idx = 0; idx <= starti; ++idx) {
        i[idx] = starti - idx;
        j[idx] = startj + idx;
    }

    int dim = count_if(j.begin(), j.end(), [N](int value) { return value < N; });

    i.resize(dim);
    j.resize(dim);

    if (reverse) {
        for (int idx = 0; idx < dim; ++idx) {
            i[idx] = M - 1 - i[idx];
            j[idx] = N - 1 - j[idx];
        }
    }

    for (int idx = 0; idx < dim; ++idx) {
        i[idx] += adjusted_box[0];
        j[idx] += adjusted_box[2];
    }

    return make_pair(i, j);
}
void update_alignment_metadata(std::map<std::string, double>& metadata, int newcells) {
    if (metadata.find("M") != metadata.end() && metadata.find("N") != metadata.end() && metadata.find("totalCells") != metadata.end()) {
        int M = metadata["M"];
        int N = metadata["N"];
        int totalCells = metadata["totalCells"];
        int denom = M * N;
        int before = static_cast<int>(floor(50.0 * totalCells / denom));
        totalCells += newcells;
        metadata["totalCells"] = totalCells;
        int after = static_cast<int>(floor(50.0 * totalCells / denom));
        int perc = 1;
        if (metadata.find("perc") != metadata.end()) {
            perc = metadata["perc"];
        }

        if (after > before && after % perc == 0) {
            std::cout << "Parallel Alignment " << after << "% ";
            if (metadata.find("timeStart") != metadata.end()) {
                auto timeStart = metadata["timeStart"];
                auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
                //转换timeStart 只保留秒
                timeStart = timeStart / 1000000000;
                //转换now 只保留秒
                now = now / 1000000000;

                auto elapsed_seconds = (now - timeStart)/1;  // 转换为秒
                std::cout << "Elapsed time: " << elapsed_seconds << "s" << std::endl;

            }
        }
    }
}


std::vector<std::pair<int, int>> refine_warping_path(const std::vector<std::pair<int, int>>& path) {
    int N = path.size();

    // Step 1: Identify all vertical and horizontal segments
    std::vector<std::tuple<std::string, int, int>> vert_horiz;
    int i = 0;
    while (i < N - 1) {
        if (path[i + 1].first == path[i].first) {
            // Vertical line
            int j = i + 1;
            while (j < N && path[j].first == path[i].first) {
                j++;
            }
            vert_horiz.push_back({"vert", i, j - 1});
            i = j;
        } else if (path[i + 1].second == path[i].second) {
            // Horizontal line
            int j = i + 1;
            while (j < N && path[j].second == path[i].second) {
                j++;
            }
            vert_horiz.push_back({"horiz", i, j - 1});
            i = j;
        } else {
            i++;
        }
    }

    // Step 2: Compute local densities
    std::vector<double> xidx;
    std::vector<double> density;
    i = 0;
    int vhi = 0;
    while (i < N) {
        int inext = i + 1;
        if (vhi < vert_horiz.size() && std::get<1>(vert_horiz[vhi]) == i) {
            auto v = vert_horiz[vhi];
            int n_seg = std::get<2>(v) - std::get<1>(v) + 1;
            std::vector<double> xidxi;
            std::vector<double> densityi;
            int n_seg_prev = 0;
            int n_seg_next = 0;
            if (vhi > 0 && std::get<2>(vert_horiz[vhi - 1]) == i) {
                n_seg_prev = std::get<2>(vert_horiz[vhi - 1]) - std::get<1>(vert_horiz[vhi - 1]) + 1;
            }
            if (vhi < vert_horiz.size() - 1 && std::get<1>(vert_horiz[vhi + 1]) == std::get<2>(v)) {
                n_seg_next = std::get<2>(vert_horiz[vhi + 1]) - std::get<1>(vert_horiz[vhi + 1]) + 1;
            }
            if (std::get<0>(v) == "vert") {
                for (int k = 0; k <= n_seg; ++k) {
                    xidxi.push_back(path[i].first + static_cast<double>(k) / n_seg);
                }
                densityi.assign(n_seg + 1, n_seg);
                if (n_seg_prev > 0) {
                    densityi[0] = static_cast<double>(n_seg) / n_seg_prev;
                }
                if (n_seg_next > 0) {
                    densityi[n_seg - 1] = static_cast<double>(n_seg) / n_seg_next;
                    densityi[n_seg] = static_cast<double>(n_seg) / n_seg_next;
                    inext = std::get<2>(v);
                } else {
                    inext = std::get<2>(v) + 1;
                }
            } else {
                for (int k = 0; k < n_seg; ++k) {
                    xidxi.push_back(path[i].first + k);
                }
                densityi.assign(n_seg, 1.0 / n_seg);
                if (n_seg_prev > 0) {
                    xidxi.erase(xidxi.begin());
                    densityi.erase(densityi.begin());
                }
                if (n_seg_next > 0) {
                    inext = std::get<2>(v);
                } else {
                    inext = std::get<2>(v) + 1;
                }
            }
            xidx.insert(xidx.end(), xidxi.begin(), xidxi.end());
            density.insert(density.end(), densityi.begin(), densityi.end());
            vhi++;
        } else {
            xidx.push_back(path[i].first);
            xidx.push_back(path[i].first + 1);
            density.push_back(1.0);
            density.push_back(1.0);
        }
        i = inext;
    }

    // Step 3: Integrate densities
    std::vector<std::pair<int, int>> path_refined = {{0, 0}};
    double j = 0.0;
    for (size_t i = 1; i < xidx.size(); ++i) {
        if (xidx[i] > xidx[i - 1]) {
            j += (xidx[i] - xidx[i - 1]) * density[i - 1];
            path_refined.push_back({static_cast<int>(xidx[i]), static_cast<int>(j)});
        }
    }

    return path_refined;
}