/**
 * @file random_generator.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-06-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <chrono>
#include <random>


class NovosibirskGenerator
{
protected:
    double m_mu, m_sigma, m_eta, m_xi, m_s0, m_xmax;
    std::mt19937 m_gen;
    std::normal_distribution<double> m_gaus;
    std::uniform_real_distribution<double> m_uni;

public:
    NovosibirskGenerator(double mu, double sigma, double eta);
    double generate();
};