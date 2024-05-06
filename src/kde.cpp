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
    double sigma = 0.2;
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
        if (abs(*it) > 3)
            continue;

        mean += *it;
        count++;
    }
    
    mean /= count;

    // Calculate dev
    double dev = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
    {
        if (abs(*it) > 3)
            continue;

        dev += pow((*it - mean), 2);
    }

    dev /= (count - 1);
    dev = sqrt(dev);

    // Calculate h
    m_h = pow(4. / (3.*count), 0.2) * dev;// / 4;
 
    // Create gaus
    auto gaus = [&](double x, double y){ return 1. / (sqrt(2 * M_PI) * m_h) * exp(-0.5*pow((x - y)/m_h, 2)); };
    auto dgaus = [&](double x, double y){ return -1. / (sqrt(2 * M_PI) * m_h) * exp(-0.5*pow((x - y)/m_h, 2)) * (x-y)/pow(m_h, 2); };

    m_kl = 0;
    m_dkl = 0;

    clock_t start, end;
    start = clock();
    m_f.reserve(reco.size());

    std::vector<double> m_qs, m_logs;
    m_qs.reserve(reco.size());
    m_logs.reserve(reco.size());

    int reco_size = reco.size();
    for (int i = 0; i < reco_size; i++)
    {
        double val = reco.at(i);
        if (abs(val) > 3)
        {
            m_f.push_back(0);
            m_qs.push_back(0);
            m_logs.push_back(0);
            continue;
        }

        double p = 0, q = m_expected_f(val);

        if (q == 0)
            q = 1e-9;

        for (auto jt = reco.begin(); jt != reco.end(); ++jt)
        {
            if (abs(*jt) > 3)
                continue;

            p += gaus(val, *jt);
        }

        p /= count;

        m_f.push_back(p);
        m_qs.push_back(q);
        m_logs.push_back(log(2*p/(p+q)));

        m_kl += 0.5 * (log(2*p/(p+q)) + q/p *log(2*q/(p+q)));
    }

    m_kl /= count;
    end = clock();
    if (m_verbose)
        std::cout << "1s part: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;

    // we can store expected_f and log(p/q) by one loop

    start = clock();
    m_grads.reserve(reco.size());
    for (int i = 0; i < reco_size; i++)
    {
        double val = reco.at(i);
        if (abs(val) > 3)
        {
            m_grads.push_back(0);
            continue;
        }

        double dp = 0;
        for (auto jt = reco.begin(); jt != reco.end(); ++jt)
        {
            if (abs(*jt) > 3)
                continue;

            int ind = std::distance(reco.begin(), jt);
            double pj = m_f.at(ind);
            double qj = m_qs.at(ind);//m_expected_f(*jt);
            if (qj == 0)
                qj = 1e-9;

            double part = dgaus(val, *jt) / pj * (m_logs.at(ind));// + 1); 
            dp += part;
        }
        dp /= count;

        m_grads.push_back(dp);

        m_dkl += dp;
    }
    end = clock();
    if (m_verbose)
        std::cout << "2nd part: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;

    if (m_verbose)
    {
        std::cout << "m: " << mean << std::endl;
        std::cout << "d: " << dev << std::endl;
        std::cout << "h: " << m_h << std::endl;
        std::cout << "kl: " << m_kl << std::endl;
        std::cout << "dkl: " << m_dkl << std::endl << std::endl;
    }
}