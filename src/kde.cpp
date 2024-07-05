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
    double sigma = 0.24;
    double xi = 2 * sqrt(2*log(2));
    double eta = 0.15;
    double s0 = 2/xi * log(xi*eta/2 + sqrt(1 + pow(xi*eta/2, 2)));
    m_expected_f = [sigma, eta, s0](double x){double part1 = 1., part2 = 1.;
                                              if ((x+1)*eta/sigma < 1)
                                                part1 = std::erf( (s0*s0 - log(1 - (x+1)*eta/sigma)) / (sqrt(2) * s0) );
                                              if((x-1)*eta/sigma < 1)
                                                part2 = std::erf( (s0*s0 - log(1 - (x-1)*eta/sigma)) / (sqrt(2) * s0) );
                                              return 0.5 * (part1 - part2);};
    m_expected_df = [sigma, eta, s0](double x){double part1 = 1., part2 = 1.;
                                               if ((x+1)*eta/sigma > 1)
                                                part1 = -exp(-0.5*pow(log(1-eta*(x+1)/sigma)/s0, 2));
                                               if ((x-1)*eta/sigma > -1) 
                                                part2 = -exp(-0.5*pow(log(1-eta*(x-1)/sigma)/s0, 2));
                                               return 0.5 * eta * exp(-0.5*s0*s0) / (sqrt(2*M_PI) * sigma * s0 ) * (part1 - part2);};
    //m_expected_f = [sigma](double x){return 0.25 * (std::erf((1-x)/(sqrt(2)*sigma)) + std::erf((1+x)/(sqrt(2)*sigma)));};
    //m_expected_df = [sigma](double x){return 0.5/(sqrt(2*M_PI) * sigma) * (exp(-0.5*pow((1+x)/sigma, 2)) - exp(-0.5*pow((1-x)/sigma, 2)));};
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
    m_h = pow(4. / (3.*count), 0.2) * dev / 4;
 
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

        for (auto jt = reco.begin(); jt != reco.end(); ++jt)
        {
            if (abs(*jt) > 3)
                continue;

            p += gaus(val, *jt);
        }

        p /= count;

        m_f.push_back(p);
        m_qs.push_back(q);
        //m_logs.push_back(log(2*p/(p+q)));
        double clog = (q == 0 || p == 0) ? 0. : log(p/q);
        //std::cout << clog << " " << val << " " << q << std::endl;
        //int k;
        //std::cin >> k;
        m_logs.push_back(clog);

        //m_kl += 0.5 * (log(2*p/(p+q)) + q/p *log(2*q/(p+q))); //Jef
        m_kl += clog;
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

            //double part = dgaus(val, *jt) / pj * (m_logs.at(ind));// + 1); // Jef
            double part = dgaus(val, *jt) / pj * (m_logs.at(ind) + 1); 
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
