/**
 * @file random_generator.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-06-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "../include/random_generator.h"


NovosibirskGenerator::NovosibirskGenerator(double mu, double sigma, double eta)
{
    m_mu = mu;
    m_sigma = sigma;
    m_eta = eta;
    m_xi = 2 * sqrt(2*log(2.));
    m_s0 = 2./m_xi * log(m_xi*m_eta/2 + sqrt(1 + pow(m_xi*m_eta/2., 2)));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    m_gen = std::mt19937(seed);
    m_gaus = std::normal_distribution<double>(mu, sigma);
    double Agauss = 1./(sqrt(2*M_PI) * m_sigma);
    double Alog = m_eta / (sqrt(2*M_PI) * m_sigma * m_s0) * exp(-0.5*m_s0*m_s0);
    double majorant = Alog / Agauss + 1e-5;
    m_uni = std::uniform_real_distribution<double>(0, majorant);
    m_xmax = m_mu + m_sigma / m_eta;
}

double NovosibirskGenerator::generate()
{
    double Agauss = 1./(sqrt(2*M_PI) * m_sigma);
    double Alog = m_eta / (sqrt(2*M_PI) * m_sigma * m_s0) * exp(-0.5*m_s0*m_s0);

    while (true)
    {
        double x = m_gaus(m_gen);
        if (x >= m_xmax)
            continue;

        double A = m_uni(m_gen);
        double ratio = Alog / Agauss * exp(-0.5*pow(log(1 - m_eta*(x - m_mu)/m_sigma)/m_s0, 2) + 0.5*pow(x/m_sigma, 2));
        if (ratio < A)
            continue;

        return x;
    }
}