#include "separability.hpp"
#include <Eigen/Dense>


// EIGEN IS AVAILABLE, NICE!
// helpers for variant 4:
bool is_separable(
    const std::vector<float>& kernel,
    int k,
    std::vector<float>& row_filter,
    std::vector<float>& col_filter,
    float tol)
{
    Eigen::MatrixXf K(k, k);

    for (int i = 0; i < k * k; i++)
        K(i / k, i % k) = kernel[i];

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(K, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXf S = svd.singularValues();

    float largest = S(0);
    float second = S.size() > 1 ? S(1) : 0.0f;

    if (second > largest * tol) {
        return false; // not rank-1
    }

    // Separable
    float sigma = std::sqrt(largest);
    Eigen::VectorXf u = svd.matrixU().col(0) * sigma;
    Eigen::VectorXf v = svd.matrixV().col(0) * sigma;

    row_filter.resize(k);
    col_filter.resize(k);

    for (int i = 0; i < k; i++) {
        col_filter[i] = u(i);
        row_filter[i] = v(i);
    }

    return true;
}


