/**
 * @file kstest.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>


class KStest
{
    bool m_verbose;
    double m_k;
    double m_deltaX;
    double m_x0;
    double m_sup;
    std::function<double(double)> m_cdf;
    std::function<double(double)> m_grad;
    std::vector<double> m_gradients;
public:
    KStest(double deltaX, std::function<double(double)> cdf);
    void calculateKS(const std::vector<double>& x);
    double get_gradient (int id) const;
    double get_sup() const {return m_sup;};
    double get_x0() const {return m_x0;};
    void set_verbose() {m_verbose = true;};
};