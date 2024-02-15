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

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>


class KMeans
{
    int m_dimension;
    int m_num;
    std::vector<std::vector<double>> m_centers;
    std::vector<int> m_num_in_cluster;

    const std::vector<std::vector<double>> calc_new_centers(const std::vector<std::vector<double>>& data);
public:
    KMeans(int num_clusters);
    void calculate(const std::vector<std::vector<double>>& data);
    std::vector<std::vector<double>> get_centers() const;
    static const double distance(const std::vector<double>& x1, const std::vector<double>& x2);
    const int get_closest_center(const std::vector<double>& item, double& distance) const;
};