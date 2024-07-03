#ifndef DTWC_ALIGNMENTTOOLS_H
#define DTWC_ALIGNMENTTOOLS_H


#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <map>
#include <chrono>

using namespace std;
using namespace std::chrono;

int get_diag_len(const vector<int>& box, int k);
pair<vector<int>, vector<int>> get_diag_indices(int MTotal, int NTotal, int k, const vector<int>& box = {}, bool reverse = false);
void update_alignment_metadata(std::map<std::string, long double>& metadata, int newcells = 0);
std::vector<std::pair<int, int>> refine_warping_path(const std::vector<std::pair<int, int>>& path);
std::vector<std::vector<float>> stretch_audio(const std::vector<float>& x1, const std::vector<float>& x2, int sr, std::vector<std::pair<int, int>>& path, int hop_length, bool refine);
#endif //DTWC_ALIGNMENTTOOLS_H
