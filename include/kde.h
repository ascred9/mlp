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
#include <thread>

#include "eigen-3.4.0/Eigen/Core"
#include "eigen-3.4.0/Eigen/Eigenvalues"

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

#include "random_generator.h"


class KDE
{
    std::vector<double> m_grads;
    bool m_verbose = false;
    double m_h;
    double m_kl;
    double m_dkl;
    std::vector<std::vector<double>> m_hist;
public:
    std::function<double(double)> m_expected_f, m_expected_df;
    std::vector<double> m_f, m_gen;
    KDE(int type = 0);
    void recalculate_exclusive(const std::vector<double>& reco);
    void fast_recalculate(const std::vector<double>& reco); // A poorer approxamitaion but fast
    void recalculate_inclusive(const std::vector<double>& data, const std::vector<double>& reco);
    double get_gradient(unsigned int id) const {return m_grads.at(id);};
    double get_kl() {return m_kl;};
    double get_dkl() {return m_dkl;};
    void set_expected_distrib(std::function<double(double)> f_expected) {m_expected_f = f_expected;};
    void set_verbose() {m_verbose = true;};
    void set_parameters(double sleft, double sright);
};