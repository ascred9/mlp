/**
 * @file kmeans.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-02-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "../include/kmeans.h"

KMeans::KMeans(int num_clusters):
    m_num(num_clusters)
    {}

void KMeans::calculate(const std::vector<std::vector<double>>& data)
{
    if (data.size() < m_num)
        throw std::invalid_argument("incorrect size of input size! it is less than number of clusters!");

    m_dimension = data.at(0).size();

}

std::vector<std::vector<double>> KMeans::get_centers() const
{
    return m_centers;;
}