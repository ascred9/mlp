/**
 * @file kstest.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "../include/kstest.h"

KStest::KStest(double deltaX, std::function<double(double)> cdf)
{
    m_deltaX = deltaX;
    m_k = 10 / m_deltaX;
    m_cdf = cdf;
    m_grad = [this](double x) {
        double arg = std::exp(-2 * m_k * (m_x0 - x));
        return arg < 5 ? -2 * m_k * arg / pow(1+arg, 2) : 0;
    };
}

void KStest::calculateKS(const std::vector<double>& x)
{
    int n = x.size();
    std::multiset<double> sorted_x = std::multiset<double>(x.begin(), x.end());

    int count = 0;
    m_sup = 0;
    for (auto xi: sorted_x)
    {
        count++;

        if (abs(m_sup) < abs(count * 1./n - m_cdf(xi)))
        {
            m_sup = count * 1./n - m_cdf(xi);
            m_x0 = xi;
        }
    }

    m_gradients.clear();
    for (auto xi: x)
    {
        //m_gradients.push_back((m_sup > 0 ? 1 : -1) * m_grad(xi));
        m_gradients.push_back(m_sup * m_grad(xi));
    }

    if (m_verbose)
    {
        std::cout << "x0: " << m_x0 << std::endl;
        std::cout << "sup: " << m_sup << std::endl;
    }
}
    
double KStest::get_gradient (int id) const{
    return m_gradients.at(id);
}