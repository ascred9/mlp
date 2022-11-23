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


#include "../include/layer_deque.h"


template class LayerDeque<Layer>;

template <class LayerType>
LayerDeque<LayerType>::LayerDeque():
    m_outsize(0),
    m_step(0.5),
    m_regulization_rate(0.)
{
}

template <class LayerType>
LayerDeque<LayerType>::~LayerDeque()
{
}

template <class LayerType>
void LayerDeque<LayerType>::add_layers(std::vector<unsigned int> topology)
{
    m_layers.reserve(topology.size());
    std::transform(begin(topology), end(topology), std::back_inserter(m_layers), [](unsigned int layer_size){
        return std::make_shared<LayerType>(layer_size);
    });
    for (auto iter = begin(m_layers); iter != end(m_layers); ++iter)
    {
        if (std::next(iter) != end(m_layers))
        {
            iter->get()->set_out_size(std::next(iter)->get()->size());
            std::next(iter)->get()->set_in_size(iter->get()->size());
        }
        else
        {
            iter->get()->set_out_size(0);
        }
        
        // default initialization of activization functions
        if (iter == begin(m_layers) || iter == end(m_layers))
            iter->get()->set_func("linear");
        else
            iter->get()->set_func("sigmoid");
    }
    m_outsize = m_layers.back()->size();
}

template <class LayerType>
std::vector<double> LayerDeque<LayerType>::calculate(const std::vector<double>& input) const
{
    if (m_layers.front()->size() != input.size())
        throw std::invalid_argument("wrong input size, it is not equal to input layer size"); // TODO: make an global Exception static class

    int size = input.size();
    Vector res( std::move(Eigen::Map<const Vector, Eigen::Aligned>(input.data(), size)) );
    for (const auto& pLayer: m_layers)
    {
        res = pLayer->calculate(res);
    }
    return std::vector<double>(res.data(), res.data() + res.size());
}

template <class LayerType>
void LayerDeque<LayerType>::clear()
{
    m_layers.clear();
}

template <class LayerType>
void LayerDeque<LayerType>::generate_weights(const std::string& init_type)
{
    for (auto& layer: m_layers)
    {
        layer->generate_weights(init_type);
    }
}

template <class LayerType>
double LayerDeque<LayerType>::get_step() const
{
    return m_step;
}

template <class LayerType>
void LayerDeque<LayerType>::print(std::ostream& os) const
{
    os << "regulization_rate " << m_regulization_rate << std::endl;
    for(const auto& pLayer: m_layers)
        pLayer->print(os);
}

template <class LayerType>
void LayerDeque<LayerType>::set_active_funcs(const std::vector<std::string>& active_funcs)
{
    if (active_funcs.size() != m_layers.size())
        throw std::invalid_argument("number of bias is not equal to number of transform matrices");

    for (unsigned int idx = 0; idx < active_funcs.size(); ++idx)
        m_layers.at(idx)->set_func(active_funcs.at(idx));
}

template <class LayerType>
void LayerDeque<LayerType>::set_layers(const std::vector<std::vector<double>>& matrices, const std::vector<std::vector<double>>& biases)
{
    if (matrices.size() != biases.size())
        throw std::invalid_argument("number of bias is not equal to number of transform matrices");

    if (m_layers.size()-1 != biases.size()) // end layer will have default init
        throw std::invalid_argument("number of layer is not equal to number of bias/transform matrices");
    
    for (unsigned int idx = 0; idx < m_layers.size()-1; ++idx)
    {
        m_layers[idx]->set_matrixW(matrices.at(idx));
        m_layers[idx]->set_vectorB(biases.at(idx));
    }
}

template <class LayerType>
void LayerDeque<LayerType>::set_loss_func(const std::string& loss_type)
{
    m_loss_type = loss_type;
    if (m_loss_type == "LS")
    {
        m_floss = [this](const Vector& real, const Vector& output){return 0.5 * ((output - real).array() * (output - real).array()) / m_outsize ;};
        m_fploss = [this](const Vector& real, const Vector& output){return ((output-real).array()) / m_outsize;};
    }
    else if (m_loss_type == "LQ")
    {
        m_floss = [this](const Vector& real, const Vector& output){return 0.125 * ((output - real).array() * (output - real).array() * 
            (output - real).array() * (output - real).array()) / m_outsize ;};
        m_fploss = [this](const Vector& real, const Vector& output){return 0.5 * ((output - real).array() * (output - real).array() *
            (output - real).array())/ m_outsize;};
    }
    else if (m_loss_type == "LSPL")
    {
        m_floss = [this](const Vector& real, const Vector& output){return 0.5 * ((output - real).array() * (output - real).array()) / m_outsize + 0.25 * (output-real).sum() / m_outsize;};
        m_fploss = [this](const Vector& real, const Vector& output){return (output-real) / m_outsize + Vector::Constant(m_outsize, 0.25) / m_outsize;};
    }
    else if (m_loss_type == "ABS")
    {
        auto magn = [](double a) {return abs(a);};
        m_floss = [this, magn](const Vector& real, const Vector& output){return (output - real).unaryExpr(magn) / m_outsize ;};
        auto sign = [](double a) {
            if ( a > 0)
                return +1.;
            else if ( a < 0)
                return -1.;
            else
                return 0.;
        };
        m_fploss = [this, sign](const Vector& real, const Vector& output){return (output-real).unaryExpr(sign) / m_outsize;};
    }
    else if (m_loss_type == "RABS")
    {
        auto magn = [](double a) {return abs(a);};
        m_floss = [this, magn](const Vector& real, const Vector& output){return (output-real).unaryExpr(magn).array() / real.array()
             / m_outsize ;};
        auto sign = [](double a) {
            if ( a > 0)
                return +1.;
            else if ( a < 0)
                return -1.;
            else
                return 0.;
        };
        m_fploss = [this, sign](const Vector& real, const Vector& output){return (output-real).unaryExpr(sign).array() / real.array()
            / m_outsize;};
    }
    else if (m_loss_type == "GOOGLE")
    {
        double a = -20.;
        double c = 200.;
        auto f = [a, c](double x) {return abs(a-2.)/a * ( pow(pow(x/c, 2) / abs(a-2.) + 1., a/2.) - 1.);};
        auto df = [a, c](double x) {return  x/(c*c) * ( pow(pow(x/c, 2) / abs(a-2.) + 1., a/2. - 1) );};
        m_floss = [this, f](const Vector& real, const Vector& output) {return (output-real).unaryExpr(f) / m_outsize; };
        m_fploss = [this, df](const Vector& real, const Vector& output) {return (output-real).unaryExpr(df) / m_outsize; };
    }
    else
        throw std::invalid_argument("this loss function isn't implemented");
}

template <class LayerType>
void LayerDeque<LayerType>::set_regulization_rate(double regulization_rate)
{
    m_regulization_rate = regulization_rate;
}

template <class LayerType>
void LayerDeque<LayerType>::set_step(const double step)
{
    m_step = step;
}

template <class LayerType>
double LayerDeque<LayerType>::test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                        const std::vector<std::vector<double>>& weights) const
{
    if (input.size() != output.size())
        throw std::invalid_argument("input size is not equal to output size of training data"); // TODO: make an global Exception static class

    if (weights.size() != output.size())
        throw std::invalid_argument("event weights size is not equal to output size of training data"); // TODO: make an global Exception static class

    double result = 0.;

    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        std::vector<double> calc = calculate(input.at(idx));
        int calc_size = calc.size();
        const Vector Ycalc = Eigen::Map<const Vector, Eigen::Aligned>(calc.data(), calc_size);
        
        int out_size = output.at(idx).size();
        const Vector Yreal = Eigen::Map<const Vector, Eigen::Aligned>(output.at(idx).data(), out_size);

        int weights_size = output.at(idx).size();
        const Vector W = Eigen::Map<const Vector, Eigen::Aligned>(weights.at(idx).data(), weights_size);

        result += (W.array() * m_floss(Yreal, Ycalc).array()).sum();

    }

    return sqrt(result / output.size() + get_L2_regulization());
}

template <class LayerType>
void LayerDeque<LayerType>::train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                       const std::vector<std::vector<double>>& weights, unsigned int batch_size)
{
    if (input.size() != output.size())
        throw std::invalid_argument("input size is not equal to output size of training data"); // TODO: make an global Exception static class

    if (weights.size() != output.size())
        throw std::invalid_argument("event weights size is not equal to output size of training data"); // TODO: make an global Exception static class
    // Reserve memory and define reset func
    std::vector<Matrix> gradients_W;
    std::vector<Vector> gradients_B;
    gradients_W.reserve(m_layers.size() - 1);
    gradients_B.reserve(m_layers.size() - 1);

    auto reset_gradients = [this, &gradients_W, &gradients_B](){
        gradients_W.clear();
        gradients_B.clear();
        for (unsigned int idl = 0; idl < m_layers.size() - 1; ++idl)
        {
           gradients_W.emplace_back( Matrix::Zero(m_layers.at(idl)->size(), m_layers.at(idl+1)->size()) );
           gradients_B.emplace_back( Vector::Zero(m_layers.at(idl+1)->size()) );
        }
    };

    reset_gradients();

    // Calculate gradients
    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        if (m_layers.front()->size() != input.at(idx).size())
            throw std::invalid_argument("wrong input size"); // TODO: make an global Exception static class

        if (m_layers.back()->size() != output.at(idx).size())
            throw std::invalid_argument("wrong output size"); // TODO: make an global Exception static class

        std::vector<std::pair<Matrix, Vector>> gradient = get_gradient(input.at(idx), output.at(idx), weights.at(idx));

        for (unsigned idl = 0; idl < m_layers.size() - 1; ++idl)
        {
            gradients_W.at(idl) += gradient.at(idl).first;
            gradients_B.at(idl) += gradient.at(idl).second;
        }

        if ( (idx + 1) % batch_size == 0)
        {
            for (unsigned int idl = 0; idl < m_layers.size() - 1; ++idl)
            {
                double weight_decay = 1. - m_step * m_regulization_rate / m_outsize;
                m_layers.at(idl)->set_matrixW(weight_decay * m_layers.at(idl)->get_matrixW() - m_step * gradients_W.at(idl) / batch_size);
                m_layers.at(idl)->set_vectorB(weight_decay * m_layers.at(idl)->get_vectorB() - m_step * gradients_B.at(idl) / batch_size);
            }
            if (input.size() - idx < batch_size) // nothing to batch
                break;

            reset_gradients();
        }
    }
}

template <class LayerType>
std::vector<std::pair<Matrix, Vector>> LayerDeque<LayerType>::get_gradient(const std::vector<double>& input, const std::vector<double>& output,
                                                                const std::vector<double>& weights) const
{
    std::vector<std::pair<Matrix, Vector>> dL;
    dL.reserve(m_layers.size()-1);

    int in_size = input.size();
    const Vector Zin = Eigen::Map<const Vector, Eigen::Aligned>(input.data(), in_size);

    int out_size = output.size();
    const Vector Y = Eigen::Map<const Vector, Eigen::Aligned>(output.data(), out_size);

    int weights_size = weights.size();
    const Vector W = Eigen::Map<const Vector, Eigen::Aligned>(weights.data(), weights_size);

    // Xi = f(sum( Wij * Zj) + Bi), i - output neuron, j - input neurons
    std::vector<std::shared_ptr<const Vector>> pZ_layers; // (Z0, Z1, Z2, ..., Zn), this vector has size m_layers.size()
    pZ_layers.reserve(m_layers.size());
    std::vector<std::shared_ptr<const Vector>> pX_layers; // (X0, X1, X2, ..., Xn), this vector has size m_layers.size()
    pX_layers.reserve(m_layers.size());

    // init Z_layers and X_layers
    {
        std::shared_ptr<const Vector> pZ_layer = std::make_shared<const Vector>(Zin);
        pZ_layers.emplace_back(pZ_layer);
        std::shared_ptr<const Vector> pX_layer = std::make_shared<const Vector>( m_layers.front()->calculateX(*pZ_layer) );
        pX_layers.emplace_back(pX_layer);
        for (unsigned int idx = 0; idx < m_layers.size() - 1; ++idx)
        {
            pZ_layer = std::make_shared<const Vector>( m_layers.at(idx)->calculateZ(*pX_layer) );
            pX_layer = std::make_shared<const Vector>( m_layers.at(idx+1)->calculateX(*pZ_layer) );

            pZ_layers.emplace_back(pZ_layer);
            pX_layers.emplace_back(pX_layer);
        }
    }

    Vector delta = (W.array() * m_fploss(Y, *pX_layers.back()).array()) * m_layers.back()->calculateXp( *pZ_layers.back() ).array();
    for (int idx = m_layers.size()-2; idx > -1; --idx)
    {
        dL.emplace_back( std::pair<Matrix, Vector>((*pX_layers.at(idx)).transpose() * delta, delta) );

        if (idx == 0)
            break;
            
        delta = m_layers.at(idx)->calculateXp( *pZ_layers.at(idx) ).array() * (delta * m_layers.at(idx)->get_matrixW().transpose()).array();
    }

    std::reverse( std::begin(dL), std::end(dL) );
    return dL;
}

template <class LayerType>
double LayerDeque<LayerType>::get_L2_regulization() const
{
    if (m_regulization_rate == 0)
        return 0;

    double regulization = 0;
    for (const auto& pLayer: m_layers)
    {
        regulization += (pLayer->get_matrixW().array() * pLayer->get_matrixW().array()).sum();
        regulization += (pLayer->get_vectorB().array() * pLayer->get_vectorB().array()).sum();
    }
    return 0.5 * m_regulization_rate * regulization / m_outsize;
}
