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
    double sigma = 0.1;
    m_expected_f = [sigma](double x){return 0.25 * (std::erf((1-x)/(sqrt(2)*sigma)) + std::erf((1+x)/(sqrt(2)*sigma)));};
    m_expected_df = [sigma](double x){return 0.5/(sqrt(2*M_PI) * sigma) * (exp(-0.5*pow((-1-x)/sigma, 2)) - exp(-0.5*pow((1-x)/sigma, 2)));};
    //m_expected_f = [sigma](double x){return 1./(sqrt(2*M_PI)*sigma) * exp(-0.5*pow(x/sigma, 2));};
}

void KDE::recalculate(const std::vector<double>& reco)
{
    m_grads.clear();
    m_f.clear();

    if (reco.size() == 0)
        throw std::invalid_argument("kde isn't implemented for size less than 1");

    // Calculate mean
    int count = 0;
    double mean = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
    {
        if (abs(*it) > 2)
            continue;

        mean += *it;
        count++;
    }
    
    mean /= count;

    // Calculate dev
    double dev = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
    {
        if (abs(*it) > 2)
            continue;

        dev += pow((*it - mean), 2);
    }

    dev /= (count - 1);
    dev = sqrt(dev);

    // Calculate h
    m_h = pow(4. / (3.*count), 0.2) * dev;
 
    // Create gaus
    auto gaus = [&](double x, double y){ return 1. / (sqrt(2 * M_PI) * m_h) * exp(-0.5*pow((x - y)/m_h, 2)); };
    auto dgaus = [&](double x, double y){ return -1. / (sqrt(2 * M_PI) * m_h) * exp(-0.5*pow((x - y)/m_h, 2)) * (x-y)/pow(m_h, 2); };

    m_kl = 0;
    m_dkl = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
    {
        if (abs(*it) > 2)
        {
            m_f.push_back(0);
            continue;
        }

        double p = 0, q = m_expected_f(*it);

        if (q == 0)
            q = 1e-9;

        for (auto jt = reco.begin(); jt != reco.end(); ++jt)
        {
            if (abs(*jt) > 2)
                continue;

            p += gaus(*it, *jt);
        }
                
        p /= count;

        m_f.push_back(p);

        m_kl += log(p/q);
    }
    m_kl /= count;

    for (auto it = reco.begin(); it != reco.end(); ++it)
    {
        if (abs(*it) > 2)
        {
            m_grads.push_back(0);
            continue;
        }

        double dp = 0;
        for (auto jt = reco.begin(); jt != reco.end(); ++jt)
        {
            if (abs(*jt) > 2)
                continue;

            double pj = m_f.at(std::distance(reco.begin(), jt));
            double qj = m_expected_f(*jt);
            if (qj == 0)
                qj = 1e-9;

            double part = dgaus(*it, *jt) / pj * (log(pj/qj) + 1); 
            dp += part;
        }
        dp /= count;

        m_grads.push_back(dp);

        m_dkl += dp;
    }

    if (m_verbose)
    {
        std::cout << "m: " << mean << std::endl;
        std::cout << "d: " << dev << std::endl;
        std::cout << "h: " << m_h << std::endl;
        std::cout << "kl: " << m_kl << std::endl;
        std::cout << "dkl: " << m_dkl << std::endl << std::endl;
    }
}