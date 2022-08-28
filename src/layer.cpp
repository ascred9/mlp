#include "layer.h"

Layer::Layer(unsigned int size):
    m_size(size)
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

    return (inputX * m_matrixW + m_vectorB * m_bias).transpose();
}

const Vector Layer::calculateX(const Vector& inputZ) const
{
    return inputZ.unaryExpr(m_f);
}

const Vector Layer::calculateXp(const Vector& inputZ) const
{
    return inputZ.unaryExpr(m_fp);
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

const Matrix& Layer::get_matrixW() const
{
    return m_matrixW;
}

const Vector& Layer::get_vectorB() const
{
    return m_vectorB;
}

void Layer::print(std::ostream &os) const
{
    if (m_out_size == 0)
        return;
    os << "[ " << m_matrixW << " ]" << std::endl;
    os << "{ " << m_vectorB << " }" << std::endl;
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

LayerDeque::LayerDeque()
{
}

LayerDeque::~LayerDeque()
{
}

void LayerDeque::add_layers(std::vector<unsigned int> topology)
{
    m_layers.reserve(topology.size());
    std::transform(begin(topology), end(topology), std::back_inserter(m_layers), [](unsigned int layer_size){return std::make_shared<Layer>(layer_size);});
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
}

std::vector<double> LayerDeque::calculate(const std::vector<double>& input) const
{
    if (m_layers.front()->size() != input.size())
        throw std::invalid_argument("wrong input size, it is not equal to input layer size"); // TODO: make an global Exception static class

    int size = input.size();
    Vector res( std::move(Eigen::Map<const Vector, Eigen::Aligned>(input.data(), size)) );
    for (const auto& layer: m_layers)
    {
        res = layer->calculate(res);
    }
    return std::vector<double>(res.data(), res.data() + res.size());
}

void LayerDeque::clear()
{
    m_layers.clear();
}

void LayerDeque::generate_weights(const std::string& init_type)
{
    for (auto& layer: m_layers)
    {
        layer->generate_weights(init_type);
    }
}

double LayerDeque::get_step() const
{
    return m_step;
}

void LayerDeque::print(std::ostream& os) const
{
    for(const auto& pLayer: m_layers)
        pLayer->print(os);
}

void LayerDeque::set_active_funcs(const std::vector<std::string>& active_funcs)
{
    if (active_funcs.size() != m_layers.size())
        throw std::invalid_argument("number of bias is not equal to number of transform matrices");

    for (unsigned int idx = 0; idx < active_funcs.size(); ++idx)
        m_layers.at(idx)->set_func(active_funcs.at(idx));
}

void LayerDeque::set_layers(const std::vector<std::vector<double>>& matrices, const std::vector<std::vector<double>>& biases)
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

void LayerDeque::set_loss_func(const std::string& loss_type)
{
    m_loss_type = loss_type;
    if (m_loss_type == "LS")
    {
        m_floss = [](const Vector& real, const Vector& output){return 0.5 * (output - real).array() * (output - real).array();};
        m_fploss = [](const Vector& real, const Vector& output){return output-real;};
    }
    else
        throw std::invalid_argument("this loss function isn't implemented");
}

void LayerDeque::set_step(const double step)
{
    m_step = step;
}

double LayerDeque::test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output, unsigned int batch_size) const
{
    if (input.size() != output.size())
        throw std::invalid_argument("input size is not equal to output size of training data"); // TODO: make an global Exception static class

    double result = 0.;
    double delta = 0.;

    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        std::vector<double> calc = calculate(input.at(idx));
        int calc_size = calc.size();
        const Vector Ycalc = Eigen::Map<const Vector, Eigen::Aligned>(calc.data(), calc_size);
        
        int out_size = output.at(idx).size();
        const Vector Yreal = Eigen::Map<const Vector, Eigen::Aligned>(output.at(idx).data(), out_size);

        delta += m_floss(Yreal, Ycalc).sum();

        if ( (idx + 1) % batch_size == 0)
        {
            result += delta / batch_size;
            
            if (input.size() - idx < batch_size) // nothing to batch
                break;

            delta = 0;
        }
    }

    return result / ( input.size() / batch_size);
}

void LayerDeque::train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output, unsigned int batch_size)
{
    if (input.size() != output.size())
        throw std::invalid_argument("input size is not equal to output size of training data"); // TODO: make an global Exception static class

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

    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        if (m_layers.front()->size() != input.at(idx).size())
            throw std::invalid_argument("wrong input size"); // TODO: make an global Exception static class

        if (m_layers.back()->size() != output.at(idx).size())
            throw std::invalid_argument("wrong output size"); // TODO: make an global Exception static class

        std::vector<std::pair<Matrix, Vector>> gradient = get_gradient(input.at(idx), output.at(idx));

        for (unsigned idl = 0; idl < m_layers.size() - 1; ++idl)
        {
            gradients_W.at(idl) += gradient.at(idl).first;
            gradients_B.at(idl) += gradient.at(idl).second;
        }

        if ( (idx + 1) % batch_size == 0)
        {
            for (unsigned int idl = 0; idl < m_layers.size() - 1; ++idl)
            {
                m_layers.at(idl)->set_matrixW(m_layers.at(idl)->get_matrixW() - (m_step / batch_size) * gradients_W.at(idl));
                m_layers.at(idl)->set_vectorB(m_layers.at(idl)->get_vectorB() - (m_step / batch_size) * gradients_B.at(idl));
            }
            if (input.size() - idx < batch_size) // nothing to batch
                break;

            reset_gradients();
        }
    }
}

std::vector<std::pair<Matrix, Vector>> LayerDeque::get_gradient(const std::vector<double>& input, const std::vector<double>& output) const
{
    std::vector<std::pair<Matrix, Vector>> dL;
    dL.reserve(m_layers.size()-1);

    int in_size = input.size();
    const Vector Zin = Eigen::Map<const Vector, Eigen::Aligned>(input.data(), in_size);

    int out_size = output.size();
    const Vector Y = Eigen::Map<const Vector, Eigen::Aligned>(output.data(), out_size);

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

    Vector delta = m_fploss(Y, *pX_layers.back()).array() * m_layers.back()->calculateXp( *pZ_layers.back() ).array();
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