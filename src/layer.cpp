/**
 * @file layer.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-09-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "../include/layer.h"


Layer::Layer(unsigned int size):
    m_size(size),
    m_in_size(0),
    m_out_size(0),
    m_bias(1),
    m_regulization_rate(0.),
    m_viscosity_rate(0.),
    m_adagrad_rate(0.)
{
}

Layer::~Layer()
{
}

const Vector Layer::calculate(const Vector& input) const
{
    if (m_out_size == 0 ) // end layer
        return calculateX(input);

    return calculateZ(calculateX(input)); // (input.unaryExpr(m_f) * m_matrixW + m_vectorB * m_bias).transpose()
}

const Vector Layer::calculateZ(const Vector& inputX) const
{
    if (m_out_size == 0 ) // end layer
        return inputX;

    return (inputX * get_matrixW() + get_vectorB() * m_bias).transpose();
}

const Vector Layer::calculateX(const Vector& inputZ) const
{
    return inputZ.unaryExpr(m_f);
}

const Vector Layer::calculateXp(const Vector& inputZ) const
{
    return inputZ.unaryExpr(m_fp);
}

const Matrix& Layer::get_matrixW() const
{
    return m_matrixW;
}

const Vector& Layer::get_vectorB() const
{
    return m_vectorB;
}

const unsigned int& Layer::size() const
{
    return m_size;
}

void Layer::set_in_size(unsigned int in_size)
{
    m_in_size = in_size;
}

void Layer::set_out_size(unsigned int out_size)
{
    m_out_size = out_size;
}

void Layer::set_func(std::string name)
{
    if (name == "linear")
    {
        m_f = [](double x){return x;};
        m_fp = [](double x){return 1;};
    }
    else if (name == "sigmoid")
    {
        m_f =[](double x){
            if (x > 0)
            {
                double em = std::exp(-x);
                return 1. / (1. + em);
            }
            else
            {
                double ep = std::exp(x);
                return ep / (1. + ep);
            }
        };
        m_fp = [](double x){
            double sigma;
            if (x > 0)
            {
                double em = std::exp(-x);
                sigma = 1. / (1. + em);
            }
            else
            {
                double ep = std::exp(x);
                sigma = ep / (1. + ep);
            }
            return sigma * (1 - sigma);
        };
    }
    else if (name == "th")
    {
        m_f =[](double x){
            double em = std::exp(-x);
            double ep = std::exp(x);
            return (ep - em) / (ep + em);
        };
        m_fp = [](double x){
            double em = std::exp(-x);
            double ep = std::exp(x);
            return 1 - pow( (ep - em) / (ep + em) , 2);
        };
    }
    else if (name == "bent")
    {
        m_f = [](double x){return (std::sqrt(std::pow(x, 2) + 1.) - 1.)/2.+ x;};
        m_fp = [](double x){return x/(2*std::sqrt(std::pow(x, 2) + 1.)) + 1.;};
    }
    else if (name == "logic")
    {
        m_f = [](double x){return x>0? 1: 0;}; // actually, x<0? 0: 1;
        m_fp = [](double x){return x!=0? 0: std::numeric_limits<double>::infinity();};
    }
}

void Layer::set_matrixW(const std::vector<double>& weights)
{
    if (weights.size() != m_size * m_out_size)
        throw std::invalid_argument("size of input matrix is not equal to size of create layer");

    std::vector<double> vec = weights;
    m_matrixW = Eigen::Map<Matrix, Eigen::Aligned>(vec.data(), m_out_size, m_size).transpose();
}

void Layer::set_matrixW(const Matrix& weights)
{
    if (weights.size() != m_size * m_out_size)
        throw std::invalid_argument("size of input matrix is not equal to size of create layer");

    m_matrixW = weights;
}

void Layer::set_vectorB(const std::vector<double>& biases)
{
    if (biases.size() != m_out_size)
        throw std::invalid_argument("size of input vector is not equal to size of create layer");

    std::vector<double> vec = biases;
    m_vectorB = Eigen::Map<Vector, Eigen::Aligned>(vec.data(), m_out_size);
}

void Layer::set_vectorB(const Vector& biases)
{
    if (biases.size() != m_out_size)
        throw std::invalid_argument("size of input vector is not equal to size of create layer");

    m_vectorB = biases;
}

void Layer::generate_weights(const std::string& init_type)
{
    if (m_out_size == 0) // end layer
        return;

    m_matrixW = Matrix::Random(m_size, m_out_size);
    m_vectorB = Vector::Random(m_out_size);
    double coef;
    if (init_type == "Xavier")
        coef = std::sqrt( 6. / (m_size + m_out_size));
    else
        coef = 1;

    m_matrixW *= coef;
    m_vectorB *= coef;
}

double Layer::get_regulization()
{
    double regulization = 0.;
    regulization += get_matrixW().array().pow(2).sum();
    regulization += get_vectorB().array().pow(2).sum();
    regulization *= 0.5;
    return regulization;
}

void Layer::add_gradient(const std::pair<Matrix, Vector>& dL)
{
    m_gradW += (1-m_viscosity_rate) * (dL.first + m_regulization_rate * get_matrixW());
    m_gradB += (1-m_viscosity_rate) * (dL.second + m_regulization_rate * get_vectorB());
}

void Layer::print(std::ostream &os) const
{
    if (m_out_size == 0)
        return;

    os << "layer " << std::endl;
    os << "[ " << m_matrixW << " ]" << std::endl;
    os << "{ " << m_vectorB << " }" << std::endl;
}

bool Layer::read(std::istream& fin)
{
    std::string data;

    int data_weights_size = m_size*m_out_size;
    std::vector<double> weights;
    weights.reserve(data_weights_size);

    int data_biases_size = m_out_size;
    std::vector<double> biases;
    biases.reserve(data_biases_size);
    
    auto read_matrix = [&](std::string before, std::vector<double>& v){
        while (fin >> data)
        {
            if (data == before)
                break;
            v.emplace_back(std::stod(data));
        }
    };

    for (int i=0; i < 2; ++i)
    {
        fin >> data;
        if (data == "[")
            read_matrix("]", weights);

        if (data == "{")
            read_matrix("}", biases);
    }

    set_matrixW(weights);
    set_vectorB(biases);

    return true;
}

void Layer::reset_layer(const std::map<std::string, double*>& learning_pars)
{
    m_gradW = Matrix::Zero(m_size, m_out_size);
    m_gradB = Vector::Zero(m_out_size);

    m_memoryW = Matrix::Zero(m_size, m_out_size);
    m_memoryB = Vector::Zero(m_out_size);

    m_regulization_rate = *learning_pars.at("regulization");
    m_viscosity_rate = *learning_pars.at("viscosity");
    m_adagrad_rate = *learning_pars.at("adagrad");
}

void Layer::update()
{
}

void Layer::update_weights(double step)
{
    m_memoryW.array() += (1.-m_adagrad_rate) * m_gradW.array().pow(2);
    m_memoryB.array() += (1.-m_adagrad_rate) * m_gradB.array().pow(2);

    m_matrixW.array() -= step * m_gradW.array() /
      (m_memoryW.array().pow(2) +
        Matrix::Ones(m_size, m_out_size).array()).sqrt();
    m_vectorB.array() -= step * m_gradB.array() /
      (m_memoryB.array().pow(2) +
        Vector::Ones(1, m_out_size).array()).sqrt();

    m_gradW *= m_viscosity_rate;
    m_gradB *= m_viscosity_rate;

    m_memoryW *= m_adagrad_rate;
    m_memoryB *= m_adagrad_rate;
}
