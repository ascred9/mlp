/**
 * @file pca.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-02-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "../include/pca.h"

PCA::PCA()
    {}

void PCA::calculateM(const std::vector<std::vector<double>>& input)
{
    int raws = input.size();
    int cols = input.at(0).size();
    Matrix X(raws, cols);
    for (int ir = 0; ir < input.size(); ir++)
        X.row(ir) = Eigen::Map<const Vector, Eigen::Aligned> (input.at(ir).data(), cols);

    Matrix W = X.transpose()*X;

    Eigen::EigenSolver<Matrix> solver(W);
    m_values = solver.eigenvalues().real();
    m_vectors = solver.eigenvectors().real();
}

const std::pair<std::vector<double>, std::vector<std::vector<double>>> PCA::calculate(const std::vector<std::vector<double>>& input)
{
    calculateM(input);
    std::vector<double> values (m_values.data(), m_values.data() + m_values.size());
    std::vector<std::vector<double>> vectors;
    for (int ic = 0; ic < m_vectors.cols(); ic++)
        vectors.push_back(std::vector<double> (m_vectors.col(ic).data(), m_vectors.col(ic).data()+m_vectors.rows()));

    std::cout << "PCA results" << std::endl;
    std::cout << "vals: " << std::endl << m_values << std::endl;
    std::cout << "vecs: " << std::endl << m_vectors << std::endl;

    return {values, vectors};
}

void PCA::transform(std::vector<double>& input) const
{
    Vector X = Eigen::Map<Vector, Eigen::Aligned> (input.data(), input.size());
    X *= m_vectors;
    for (int ir = 0; ir < X.rows(); ir++)
        input.at(ir) = X(0, ir);
}