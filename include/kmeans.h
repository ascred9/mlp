/**
 * @file kmeans.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-02-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <stdexcept>
#include <vector>


class KMeans
{
    int m_dimension;
    int m_num;
    std::vector<std::vector<double>> m_centers;
public:
    KMeans(int num_clusters);
    void calculate(const std::vector<std::vector<double>>& data);
    std::vector<std::vector<double>> get_centers() const;
};