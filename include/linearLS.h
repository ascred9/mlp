/**
 * @file linearLS.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-04-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>


class LinearLS
{
    LinearLS();
    static bool m_verbose;
public:
    static std::array<double, 4> calculateKB(const std::vector<double>& x, const std::vector<double>& y);
};