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
    m_centers.clear();
    m_centers.reserve(m_num);
    m_num_in_cluster.assign(m_num, 0);

    // Randomize preliminary centers
    // Use random points of data as initial
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size()-1);

    std::set<int> used_id;
    while (m_centers.size() < m_num)
    {
        int id = dis(gen);
        if (used_id.find(id) != used_id.end())
            continue;

        used_id.insert(id);
        m_centers.push_back(data.at(id));
    }

    const double epsilon = 1e-4;

    int round = 0;
    clock_t start, end;
    while (true)
    {
        std::cout << "Round #" << round << std::endl;
        start = clock();
        auto new_centers = calc_new_centers(data);
        if (std::equal(m_centers.begin(), m_centers.end(), new_centers.begin(), [epsilon](const auto& vec1, const auto& vec2){
            return distance(vec1, vec2) < epsilon;
        }))
            break;

        m_centers = new_centers;
        end = clock();
        std::cout << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << "sec" << std::endl;
        ++round;
    }
}

const double KMeans::distance(const std::vector<double>& x1, const std::vector<double>& x2)
{
    if (x1.size() != x2.size())
        throw std::invalid_argument("incorrect sizes of input vecs!");

    double distsqr = 0;
    for (int i = 0; i < x1.size(); i++)
        distsqr += pow(x1.at(i) - x2.at(i), 2);
    
    return sqrt(distsqr);
}

const int KMeans::get_closest_center(const std::vector<double>& item, double& distance) const
{
    distance = std::numeric_limits<double>::infinity();
    int c_id = 0;
    for (int ic = 0; ic < m_centers.size(); ++ic)
    {
        auto cur_dist = KMeans::distance(m_centers.at(ic), item);
        if (cur_dist < distance)
        {
            distance = cur_dist;
            c_id = ic;
        }
    }

    return c_id;
}

const std::vector<std::vector<double>> KMeans::calc_new_centers(const std::vector<std::vector<double>>& data)
{
    std::vector<std::vector<double>> centers(m_num, std::vector<double>(m_dimension, 0));

    m_num_in_cluster.assign(m_num, 0);

    for (const auto& item: data)
    {
        double dist;
        int c_id = get_closest_center(item, dist);
        m_num_in_cluster.at(c_id)++;
        std::transform(centers.at(c_id).begin(), centers.at(c_id).end(), item.begin(), centers.at(c_id).begin(), [](double c, double x){
            c += x;
            return c;
        });
    }

    for (int i = 0; i < centers.size(); ++i)
        std::transform(centers.at(i).begin(), centers.at(i).end(), centers.at(i).begin(), [this, i](double val){return val / m_num_in_cluster.at(i);});

    return centers;
}

std::vector<std::vector<double>> KMeans::get_centers() const
{
    return m_centers;;
}