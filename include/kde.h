/**
 * @file kde.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-03-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "eigen-3.4.0/Eigen/Core"
#include "eigen-3.4.0/Eigen/Eigenvalues"

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

class KDE
{
    std::vector<double> m_vals;
    bool m_verbose = true;
public:
    std::function<double(double)> m_expected_f, m_expected_df;
    std::vector<double> m_f;
    KDE();
    void recalculate(const std::vector<std::vector<double>>& reco);
    const double get_val(unsigned int id) const {return m_vals.at(id);};
    void set_expected_distrib(std::function<double(double)> f_expected) {m_expected_f = f_expected;};
    void set_expected_dev_distrib(std::function<double(double)> df_expected) {m_expected_df = df_expected;};
};