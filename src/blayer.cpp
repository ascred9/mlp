/**
 * @file bayesian_layer.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-09-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../include/blayer.h"

BayesianLayer::~BayesianLayer()
{
}

const Matrix& BayesianLayer::get_matrixW() const
{
    if (m_trainMode)
        return m_tempMatrixW;
    else
        return m_matrixW;
}

const Vector& BayesianLayer::get_vectorB() const
{
    if (m_trainMode)
        return m_tempVectorB;
    else
        return m_vectorB;
}

void BayesianLayer::set_devMatrixW(const std::vector<double>& devWeights)
{
    if (devWeights.size() != m_size * m_out_size)
        throw std::invalid_argument("size of input matrix is not equal to size of create layer");

    std::vector<double> vec = devWeights;
    m_devMatrixW = Eigen::Map<Matrix, Eigen::Aligned>(vec.data(), m_out_size, m_size).transpose();
}

void BayesianLayer::set_devMatrixW(const Matrix& devWeights)
{
    if (devWeights.size() != m_size * m_out_size)
        throw std::invalid_argument("size of input matrix is not equal to size of create layer");

    m_devMatrixW = devWeights;
}

void BayesianLayer::set_devVectorB(const std::vector<double>& devBiases)
{
    if (devBiases.size() != m_out_size)
        throw std::invalid_argument("size of input vector is not equal to size of create layer");

    std::vector<double> vec = devBiases;
    m_devVectorB = Eigen::Map<Vector, Eigen::Aligned>(vec.data(), m_out_size);
}

void BayesianLayer::set_devVectorB(const Vector& devBiases)
{
    if (devBiases.size() != m_out_size)
        throw std::invalid_argument("size of input vector is not equal to size of create layer");

    m_devVectorB = devBiases;
}

void BayesianLayer::print(std::ostream& os) const 
{
    if (m_out_size == 0)
        return;

    Layer::print(os);
    os << "stddev" << std::endl;
    os << "[ " << m_devMatrixW << " ]" << std::endl;
    os << "{ " << m_devVectorB << " }" << std::endl;
    return;
}

bool BayesianLayer::read(std::istream& fin)
{
    Layer::read(fin);

    std::string data;
    auto read_matrix = [&](std::string before, std::vector<double>& v){
        while (fin >> data)
        {
            if (data == before)
                break;
            v.emplace_back(std::stod(data));
        }
    };

    if (fin >> data; data != "stddev")
        return false;

    int data_weights_size = m_size*m_out_size;
    std::vector<double> devWeights;
    devWeights.reserve(data_weights_size);

    int data_biases_size = m_out_size;
    std::vector<double> devBiases;
    devBiases.reserve(data_biases_size);

    for (int i=0; i < 2; ++i)
    {
        fin >> data;
        if (data == "[")
            read_matrix("]", devWeights);

        if (data == "{")
            read_matrix("}", devBiases);
    }

    set_devMatrixW(devWeights);
    set_devVectorB(devBiases);

    return true;
}

void BayesianLayer::generate_weights(const std::string& init_type)
{
    Layer::generate_weights(init_type);
    m_devMatrixW = 0.3 * m_matrixW.array().abs();
    m_devVectorB = 0.3 * m_vectorB.array().abs();
}

void BayesianLayer::add_gradient(const std::pair<Matrix, Vector>& dL)
{
    Layer::add_gradient(dL);
    m_gradDW.array() += (1-m_viscosity_rate) * m_epsilonW.array() * (dL.first + m_regulization_rate * get_matrixW()).array(); 
    m_gradDB.array() += (1-m_viscosity_rate) * m_epsilonB.array() * (dL.second + m_regulization_rate * get_vectorB()).array();
}


void BayesianLayer::reset_layer(const std::map<std::string, double*>& learning_pars)
{
    Layer::reset_layer(learning_pars);
    m_gradDW = Matrix::Zero(m_size, m_out_size);
    m_gradDB = Vector::Zero(m_out_size);

    m_memoryDW = Matrix::Zero(m_size, m_out_size);
    m_memoryDB = Vector::Zero(m_out_size);
}

void BayesianLayer::update()
{
    if (m_out_size == 0)
        return;

    // Generate random matrix epsilon for W and update tempMatrixW
    std::vector<double> vW(m_size * m_out_size);
    std::generate(vW.begin(), vW.end(), [&]() {return m_gaus(m_gen);});

    m_epsilonW = Eigen::Map<Matrix, Eigen::Aligned>(vW.data(), m_out_size, m_size).transpose();
    m_tempMatrixW = m_matrixW.array()
        + m_epsilonW.array() * m_devMatrixW.array();

    // Generate random matrix epsilon for B and update tempMatrixB
    std::vector<double> vB(m_out_size);
    std::generate(vB.begin(), vB.end(), std::bind(m_gaus, m_gen));

    m_epsilonB = Eigen::Map<Vector, Eigen::Aligned>(vB.data(), m_out_size);
    m_tempVectorB = m_vectorB.array()
        + m_epsilonB.array() * m_devVectorB.array();
}

void BayesianLayer::update_weights(double step)
{
    m_memoryW.array() += (1.-m_adagrad_rate) * m_gradW.array().pow(2);
    m_memoryB.array() += (1.-m_adagrad_rate) * m_gradB.array().pow(2);
    m_memoryDW.array() +=(1.-m_adagrad_rate) *m_gradDW.array().pow(2);
    m_memoryDB.array() +=(1.-m_adagrad_rate) *m_gradDB.array().pow(2);

    m_matrixW.array() -= step * m_gradW.array() /
      (m_memoryW.array().pow(2) +
        Matrix::Ones(m_size, m_out_size).array()).sqrt();
    m_vectorB.array() -= step * m_gradB.array() /
      (m_memoryB.array().pow(2) +
        Vector::Ones(1, m_out_size).array()).sqrt();

    m_devMatrixW.array() -= step * (m_gradDW.array()) /
      (m_memoryDW.array().pow(2) +
        Matrix::Ones(m_size, m_out_size).array()).sqrt();
    m_devVectorB.array() -= step * (m_gradDB.array()) /
      (m_memoryDB.array().pow(2) +
        Vector::Ones(1, m_out_size).array()).sqrt();

    auto positive = [](double a){return a > 0? a: -a;};
    m_devMatrixW = m_devMatrixW.unaryExpr(positive);
    m_devVectorB = m_devVectorB.unaryExpr(positive);

    m_gradW *= m_viscosity_rate;
    m_gradB *= m_viscosity_rate;
    m_gradDW *= m_viscosity_rate;
    m_gradDB *= m_viscosity_rate;

    m_memoryW *= m_adagrad_rate;
    m_memoryB *= m_adagrad_rate;
    m_memoryDW *= m_adagrad_rate;
    m_memoryDB *= m_adagrad_rate;
}
