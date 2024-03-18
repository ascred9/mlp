/**
 * @file pca.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-02-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "eigen-3.4.0/Eigen/Core"
#include "eigen-3.4.0/Eigen/Eigenvalues"

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

class PCA
{
    Vector m_values;
    Matrix m_vectors;
    void calculateM(const std::vector<std::vector<double>>& input);
public:
    PCA();
    const std::pair<std::vector<double>, std::vector<std::vector<double>>> calculate(const std::vector<std::vector<double>>& input);
    void transform(std::vector<double>& input) const;
};