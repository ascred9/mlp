/**
 * @file kde.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-03-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "../include/kde.h"


KDE::KDE()
{
    double sigma = 0.3;
    m_expected_f = [sigma](double x){return 0.25 * (std::erf((1-x)/(sqrt(2)*sigma)) - std::erf((-1-x)/(sqrt(2)*sigma)));};
    m_expected_df = [sigma](double x){return 0.5/(sqrt(2*M_PI) * sigma) * (exp(-0.5*pow((-1-x)/sigma, 2)) - exp(-0.5*pow((1-x)/sigma, 2)));};
}

void KDE::recalculate(const std::vector<std::vector<double>>& reco)
{
    m_vals.clear();
    m_f.clear();

    if (reco.size() < 2 && reco.at(0).size() != 1)
        throw std::invalid_argument("kde isn't implemented for size more than 1");

    // Calculate mean
    double mean = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
        mean += it->front();
    
    mean /= reco.size();

    // Calculate dev
    double dev = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
        dev += pow((it->front() - mean), 2);

    dev /= (reco.size() - 1);
    dev = sqrt(dev);

    // Calculate h
    double h = pow(4. / (3.*reco.size()), 0.2) * dev;
    
 
    // Creat gaus
    auto gaus = [h](double x, double y){ return 1. / sqrt(2 * M_PI) * exp(-0.5*pow((x - y)/h, 2)); };
    auto dgaus = [h](double x, double y){ return -1. / sqrt(2 * M_PI) * exp(-0.5*pow((x - y)/h, 2)) * (x-y)/pow(h, 2); };

    double kl = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
    {
        double p = 0, q = m_expected_f(it->front());
        double dp = 0, dq = m_expected_df(it->front());
        for (auto jt = reco.begin(); jt != reco.end(); ++jt)
        {
            p += gaus(jt->front(), it->front());
            dp += dgaus(jt->front(), it->front());
        }

        p /= (h * reco.size());
        dp /= (h * reco.size());

        m_f.push_back(p);

        m_vals.push_back(dp/p - dq/q);

        kl += log(p/q);
    }
    kl /= reco.size();
    
    if (m_verbose)
    {
        std::cout << "m: " << mean << std::endl;
        std::cout << "d: " << dev << std::endl;
        std::cout << "h: " << h << std::endl;
        std::cout << "kl: " << kl << std::endl << std::endl;
    }
}