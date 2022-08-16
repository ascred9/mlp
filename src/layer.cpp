#include "layer.h"

Layer::Layer(unsigned int size):
    m_size(size),
    m_in_size(0),
    m_out_size(0)
{
    m_bias = 0.5;
}

Layer::~Layer()
{
    std::cout << "Delete layer" << this << " with size " << m_size << std::endl;
}

const Vector Layer::calculate(const Vector& input) const
{
    if (m_out_size == 0 ) // end layer
        return input;

    return (input * m_matrixW + Vector::Constant(m_out_size, m_bias)).unaryExpr(m_f).transpose();
}

void Layer::generate_weights(const std::string& init_type)
{
    if (m_out_size == 0) // end layer
        return;

    m_matrixW = Matrix::Random(m_size, m_out_size);
    double coef;
    if (init_type == "Xavier")
        coef = std::sqrt( 6. / (m_size + m_out_size));
    else
        coef = 1;

    m_matrixW *= coef;
    m_bias = coef;
}

void Layer::print(std::ostream &os) const
{
    os << m_size << std::endl;
    os << m_matrixW << std::endl;
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
}

const unsigned int& Layer::size() const
{
    return m_size;
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
        
        if (iter == begin(m_layers) || iter == end(m_layers))
            iter->get()->set_func("linear");
        else
            iter->get()->set_func("sigmoid");
    }
}

std::vector<double> LayerDeque::calculate(const std::vector<double>& input) const
{
    if (m_layers.at(0)->size() != input.size())
        throw std::invalid_argument("wrong input"); // TODO: make an global Exception static class

    int size = input.size();
    std::vector<double> v = input;

    Vector res = Eigen::Map<Vector, Eigen::Unaligned>(v.data(), size);
    std::cout << res << " input " << std::endl;
    for (int idx=0; idx < m_layers.size() - 1; ++idx)
    {
        //std::cout << m_layers.at(idx)->calculate(res) << ", idx: " << idx << std::endl;
        res = m_layers.at(idx)->calculate(res);
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

void LayerDeque::print(std::ostream& os) const
{
    for(const auto& pLayer: m_layers)
    {
        pLayer->print(os);
        os << std::endl;;
    }
}