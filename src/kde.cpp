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


KDE::KDE(int type)
{
    double sigma = 0.2;
    double xi = 2 * sqrt(2*log(2));
    double eta = 1.73e-01;
    double s0 = 2/xi * log(xi*eta/2 + sqrt(1 + pow(xi*eta/2, 2)));
    m_expected_f = [sigma, eta, s0](double x){double part1 = 1., part2 = 1.;
                                              if ((x+1)*eta/sigma < 1)
                                                part1 = std::erf( (s0*s0 - log(1 - (x+1)*eta/sigma)) / (sqrt(2) * s0) );
                                              if((x-1)*eta/sigma < 1)
                                                part2 = std::erf( (s0*s0 - log(1 - (x-1)*eta/sigma)) / (sqrt(2) * s0) );
                                              return 0.25 * (part1 - part2);}; // where I loss 2?
    m_expected_df = [sigma, eta, s0](double x){double part1 = 0., part2 = 0.;
                                               if ((x+1)*eta/sigma < 1)
                                                part1 = -exp(-0.5*pow(log(1-eta*(x+1)/sigma)/s0, 2));
                                               if ((x-1)*eta/sigma < 1) 
                                                part2 = -exp(-0.5*pow(log(1-eta*(x-1)/sigma)/s0, 2));
                                               return 0.5 * eta * exp(-0.5*s0*s0) / (sqrt(2*M_PI) * sigma * s0 ) * (part1 - part2);};
    if (type == 0)
        //m_expected_f = [sigma](double x){return 0.25 * (std::erf((1-x)/(sqrt(2)*sigma)) + std::erf((1+x)/(sqrt(2)*sigma)));};

    if (type == 1)
        m_expected_f = [sigma](double x){return 1./(sqrt(2*M_PI)*sigma) * exp(-0.5*pow(x/sigma, 2));};
        
    //m_expected_df = [sigma](double x){return 0.5/(sqrt(2*M_PI) * sigma) * (exp(-0.5*pow((1+x)/sigma, 2)) - exp(-0.5*pow((1-x)/sigma, 2)));};


    //omp_set_dynamic(0);
    //int num = omp_get_max_threads();
    //std::cout << "Number of threads: " << num << std::endl;
    //omp_set_num_threads(num);
}

void KDE::recalculate_exclusive(const std::vector<double>& reco)
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
        //if (abs(*it) > 3)
        //    continue;

        mean += *it;
        count++;
    }
    
    mean /= count;

    // Calculate dev
    double dev = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
    {
        //if (abs(*it) > 3)
        //    continue;

        dev += pow((*it - mean), 2);
    }

    dev /= (count - 1);
    dev = sqrt(dev);

    // Calculate h
    m_h = pow(4. / (3.*count), 0.2) * dev;
 
    // Create gaus
    auto gaus = [&](double x, double y){ return 1. / (sqrt(2 * M_PI) * m_h * dev) * exp(-0.5*pow((x - y)/(m_h * dev), 2)); };
    auto dgaus = [&](double x, double y){ return -1. / (sqrt(2 * M_PI) * m_h * dev) * exp(-0.5*pow((x - y)/(m_h * dev), 2)) * (x-y)/pow(m_h * dev, 2); };

    m_kl = 0;
    m_dkl = 0;

    clock_t start, end;
    start = clock();
    m_f.resize(reco.size(), 0);

    std::vector<double> m_qs(reco.size(), 0), m_logs(reco.size(), 0);

    int reco_size = reco.size();
    //#pragma omp parallel for shared(m_f, m_qs, m_logs, reco) private(i)
    for (int i = 0; i < reco_size; i++)
    {
        double val = reco.at(i);
        //if (abs(val) > 3)
        //    continue;

        double p = 0, q = m_expected_f(val);

        for (auto jt = reco.begin(); jt != reco.end(); ++jt)
        {
            if (abs(*jt - val) > 3)
                continue;

            p += gaus(val, *jt);
        }

        p /= count;

        m_f.at(i) = p;
        m_qs.at(i) = q;
        //double clog = log(2*p/(p+q));
        double clog = (q == 0 || p == 0) ? 0. : log(p/q);
        if (q == 0 && p != 0)
            clog = 1e5;
        else if (q != 0 && p == 0)
            clog = -1e5;

        //std::cout << clog << " " << val << " " << q << std::endl;
        //int k;
        //std::cin >> k;
        m_logs.at(i) = clog;

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
        //if (abs(val) > 3)
        //{
        //    m_grads.push_back(0);
        //    continue;
        //}

        double dp = 0;
        for (auto jt = reco.begin(); jt != reco.end(); ++jt)
        {
            if (abs(*jt - val) > 5)
                continue;

            int ind = std::distance(reco.begin(), jt);

            double pj = m_f.at(ind);
            //double qj = m_qs.at(ind);//m_expected_f(*jt);

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

void KDE::recalculate_inclusive(const std::vector<double>& data, const std::vector<double>& reco)
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
        //if (abs(*it) > 3)
        //    continue;

        mean += *it;
        count++;
    }
    
    mean /= count;

    // Calculate dev
    double dev = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
    {
        //if (abs(*it) > 3)
        //    continue;

        dev += pow((*it - mean), 2);
    }

    dev /= (count - 1);
    dev = sqrt(dev);

    // Calculate h
    m_h = pow(4. / (3.*count), 0.2) * dev * 0.25;
 
    // Create gaus
    //auto gaus = [&](double x, double y){ return 1. / (sqrt(2 * M_PI) * m_h * dev) * exp(-0.5*pow((x - y)/(m_h * dev), 2)); };
    //auto dgaus = [&](double x, double y){ return -1. / (sqrt(2 * M_PI) * m_h * dev) * exp(-0.5*pow((x - y)/(m_h * dev), 2)) * (x-y)/pow(m_h * dev, 2); };

    auto gaus = [&](double x, double y){ return 1 - 0.5 * pow((x - y)/(m_h*dev), 2) > 0 ? 1. / (sqrt(2 * M_PI) * m_h * dev) * (1 - 0.5 * pow((x - y)/(m_h*dev), 2)) : 0;};
    auto dgaus = [&](double x, double y){ return 1 - 0.5 * pow((x - y)/(m_h*dev), 2) > 0 ? -(x-y)/pow(m_h * dev, 2) : 0;};

    m_kl = 0;
    m_dkl = 0;

    clock_t start, end;
    start = clock();
    m_f.resize(data.size(), 0);

    m_gen.clear();
    NovosibirskGenerator generator(0, 0.1, 0.17);

    int reco_size = reco.size();
    int data_size = data.size();
    for (int i = 0; i < data_size; i++)
    {
        double val = data.at(i) + generator.generate();
        m_gen.push_back(val);

        double p = 0;//, q = m_expected_f(val);

        for (auto jt = reco.begin(); jt != reco.end(); ++jt)
        {
            if (abs(*jt - val) > 3)
                continue;

            p += gaus(val, *jt);
        }

        p /= count;
        m_f.at(i) = p;
        //if (p == 0)
        //    std::cout << val << " " << p << std::endl;

        double clog = p == 0 ? -1e+5 : log(p);
        m_kl -= clog;
    }

    m_kl /= count;
    end = clock();
    if (m_verbose)
        std::cout << "1s part: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;

    start = clock();
    m_grads.reserve(data.size());
    for (int i = 0; i < reco_size; i++)
    {
        double val = reco.at(i);

        double dp = 0;
        for (auto j = 0; j < data_size; ++j)
        {
            double valx = m_gen.at(j);
            if (abs(valx - val) > 5)
                continue;

            double pj = m_f.at(j);
            double part = pj != 0 ? -dgaus(val, valx) / pj : (val - valx);// * 1e5;
            dp += part;

            //if (valx > 0.94 && val > 0.9)
            //    std::cout << valx << " " << val << " " << pj << " " << part << std::endl;
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

void KDE::fast_recalculate(const std::vector<double>& reco)
{
    m_grads.clear();
    m_f.clear();

    if (reco.size() == 0)
        throw std::invalid_argument("kde isn't implemented for size less than 1");

    double min = reco.front();
    double max = reco.front();

    // Calculate mean
    int count = 0;
    double mean = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
    {
        mean += *it;
        count++;

        if (min > *it)
            min = *it;

        if (max < *it)
            max = *it;
    }
    
    mean /= count;

    // Calculate dev
    double dev = 0;
    for (auto it = reco.begin(); it != reco.end(); ++it)
    {

        dev += pow((*it - mean), 2);
    }

    dev /= (count - 1);
    dev = sqrt(dev);

    // Calculate h
    m_h = pow(4. / (3.*count), 0.2) * dev;
    
    // Make a square binning
    int gate = 3;
    int nbins = std::ceil((max - min) / m_h);
    m_hist.clear();
    m_hist.resize(nbins);
    std::for_each(m_hist.begin(), m_hist.end(), [](auto& v){v.clear();});

    for (int i = 0; i < reco.size(); i++)
    {
        int ibin = (reco.at(i) - min) / m_h;
        m_hist.at(ibin).push_back(i);
    }
 
    // Create gaus
    auto gaus = [&](double x, double y){ return 1. / (sqrt(1 * M_PI) * m_h * dev) * exp(-0.5*pow((x - y)/(m_h * dev), 2)); };
    auto dgaus = [&](double x, double y){ return -1. / (sqrt(1 * M_PI) * m_h * dev) * exp(-0.5*pow((x - y)/(m_h * dev), 2)) * (x-y)/pow(m_h * dev, 2); };

    m_kl = 0;
    m_dkl = 0;

    clock_t start, end;
    start = clock();
    m_f.resize(reco.size(), 0);

    std::vector<double> m_qs(reco.size(), 0), m_logs(reco.size(), 0);

    int reco_size = reco.size();
    for (int i = 0; i < reco_size; i++)
    {
        double val = reco.at(i);

        double p = 0, q = m_expected_f(val);
        int bin = (val - min) / m_h;
        for (int ibin = 0; ibin < nbins; ibin++)
        {
            if ( abs(bin - ibin) <= gate)
            {
                for (const auto& il: m_hist.at(ibin))
                    p += gaus(val, reco.at(il));
            }
            else
            {
                p += m_hist.at(ibin).size() * gaus(val, (min + (ibin + 0.5) * m_h));
            }
        }

        p /= count;

        m_f.at(i) = p;
        double clog = (q == 0 || p == 0) ? 0. : log(p/q);
        m_logs.at(i) = clog;

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

        double dp = 0;
        int bin = (val - min) / m_h;
        for (int ibin = 0; ibin < nbins; ibin++)
        {
            if ( abs(bin - ibin) < gate)
            {
                for (const auto& il: m_hist.at(ibin))
                {
                    dp += dgaus(val, reco.at(il)) / m_f.at(il) * (m_logs.at(il) + 1);
                }
            }
            else
            {
                if (m_hist.at(ibin).size() == 0)
                    continue;

                double p = m_f.at(m_hist.at(ibin).front());
                double lg = m_logs.at(m_hist.at(ibin).front());
                dp += m_hist.at(ibin).size() * dgaus(val, (min + (ibin + 0.5) * m_h)) / p * (lg + 1);
            }
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
