#pragma once
#include <vector>

bool is_separable(
    const std::vector<float>& kernel,
    int k,
    std::vector<float>& row_filter,
    std::vector<float>& col_filter,
    float tol = 1e-5f);