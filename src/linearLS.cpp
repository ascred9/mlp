/**
 * @file linearLS.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-04-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "../include/linearLS.h"

bool LinearLS::m_verbose = false;

std::array<double, 4> LinearLS::calculateKB(const std::vector<double>& x, const std::vector<double>& y)
{
    if (x.size() != y.size())
        throw std::invalid_argument("linearLS: different input sizes");

    int n = x.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    double X, Y;

    for (int i = 0; i < n; i++)
    {
        X = x.at(i);
        Y = y.at(i);

        sumX += X;
        sumY += Y;
        sumXY += X*Y;
        sumX2 += X*X;
    }
    
    sumX = 0;
    
    double k = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    double b = (sumY - k * sumX) / n;

    if (m_verbose)
    {
        std::cout << "k: " << k << " b: " << b << " n: " << n << std::endl;
        std::cout << "x: " << sumX << " x2: " << sumX2 << std::endl;
        std::cout << "y: " << sumY << " xy: " << sumXY << std::endl;
        int kk;
        std::cin >> kk;
    }

    return {k, b, sumX, sumX2};
}