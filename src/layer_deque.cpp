/**
 * @file layer_deque.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-09-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "../include/layer_deque.h"


template
void LayerDeque::add_layers<Layer>(std::vector<unsigned int> topology);

template
void LayerDeque::add_layers<BayesianLayer>(std::vector<unsigned int> topology);

template
void LayerDeque::add_layers<GradientLayer>(std::vector<unsigned int> topology);

LayerDeque::LayerDeque():
    m_outsize(0),
    m_step(0.5),
    m_adagrad_rate(0.),
    m_dropout_rate(0.),
    m_regulization_rate(0.),
    m_viscosity_rate(0.)
{
    m_pars_map["regulization"] = &m_regulization_rate;
    m_pars_map["viscosity"] = &m_viscosity_rate;
    m_pars_map["adagrad"] = &m_adagrad_rate;
    m_pars_map["dropout"] = &m_dropout_rate;
}

LayerDeque::~LayerDeque()
{
}

template <class LayerT>
typename std::enable_if<std::is_base_of<Layer, LayerT>::value, void>::type
LayerDeque::add_layers(std::vector<unsigned int> topology)
{
    m_layers.reserve(topology.size());
    std::transform(begin(topology), end(topology), std::back_inserter(m_layers), [](unsigned int layer_size){
        return std::make_shared<LayerT>(layer_size);
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
    m_ema = Vector::Zero(m_outsize);
    m_addition_gradient = Vector::Zero(m_outsize);
}

std::vector<double> LayerDeque::calculate(const std::vector<double>& input) const
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
    os << "adagrad_rate " << m_adagrad_rate << std::endl;
    os << "dropout_rate " << m_dropout_rate << std::endl;
    os << "regulization_rate " << m_regulization_rate << std::endl;
    os << "viscosity_rate " << m_viscosity_rate << std::endl;
    for(const auto& pLayer: m_layers)
        pLayer->print(os);
}

bool LayerDeque::read_layer(std::istream& fin, int layer_id)
{
    return m_layers.at(layer_id)->read(fin);
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
        m_floss = [this](const Vector& real, const Vector& output){return (0.5 * (output - real).array().pow(2))/ m_outsize;};
        m_fploss = [this](const Vector& real, const Vector& output){return ((output-real).array()) / m_outsize;};
    }
    else if (m_loss_type == "ALS")
    {
        double A = 2.;
        auto assym_sqr = [A](double a) {
            return a > 0 ? 0.5*a*a : 0.5*a*a/A;
        };
        auto dassym_sqr = [A](double a) {
            return a > 0 ? a : a/A;
        };

        m_floss = [this, assym_sqr](const Vector& real, const Vector& output){return (output - real).array().unaryExpr(assym_sqr)/ m_outsize;};
        m_fploss = [this, dassym_sqr](const Vector& real, const Vector& output){return (output-real).array().unaryExpr(dassym_sqr) / m_outsize;};
    }

    else if (m_loss_type == "LOG")
    {
        double l = 1.;
        m_floss = [this, l](const Vector& real, const Vector& output){return ((0.5/(l*l) * (output - real).array().pow(2)) - 
            0.5 * ((output-real).array().pow(2) / (l*l)).log())/ m_outsize;};
        m_fploss = [this, l](const Vector& real, const Vector& output){return ((output-real).array()/(l*l)
            - (output-real).array().inverse()) / m_outsize;};

    }
    else if (m_loss_type == "LSshift")
    {
        m_floss = [this](const Vector& real, const Vector& output){return (0.5 * ((output - real).array().pow(2)) + 2*(output-real).array().abs()) / m_outsize;};
        m_fploss = [this](const Vector& real, const Vector& output){return (output-real).unaryExpr([](double v){return v + 2*v/abs(v);}) / m_outsize;};
    }
    else if (m_loss_type == "CORR")
    {
        m_floss = [this](const Vector& real, const Vector& output){
            double alpha = 0.99;
            static Vector ema_xy = Vector::Zero(real.size());
            static Vector ema_x = Vector::Zero(real.size());
            static Vector ema_y = Vector::Zero(real.size());
            ema_xy = alpha * ema_xy.array() + (1-alpha) * (output-real).array()*real.array();
            ema_x = alpha * ema_x + (1-alpha) * (output-real);
            ema_y = alpha * ema_y + (1-alpha) * real;
            return (ema_xy.array() - ema_x.array()*ema_y.array()).abs() / m_outsize;
        };
        m_fploss = [this](const Vector& real, const Vector& output){
            double alpha = 0.99;
            static Vector ema_xy = Vector::Zero(real.size());
            static Vector ema_x = Vector::Zero(real.size());
            static Vector ema_y = Vector::Zero(real.size());
            ema_xy = alpha * ema_xy.array() + (1-alpha) * (output-real).array()*real.array();
            ema_x = alpha * ema_x + (1-alpha) * (output-real);
            ema_y = alpha * ema_y + (1-alpha) * real;
            Vector corr = (ema_xy.array() - ema_x.array()*ema_y.array()).unaryExpr([](double v){return v > 0? 1.: -1.;});
            //std::cout << ema_xy - ema_x*ema_y << " " << ema_xy << " " << ema_x << " " << ema_y << std::endl;
            Vector df = corr.array() * (1-alpha) * (real-ema_y).array() / m_outsize;
            return df;
        };
    }
    else if (m_loss_type == "ABSO")
    {
        m_floss = [this](const Vector& real, const Vector& output){return (0.5 * ((output - real).array().pow(2) - 1).abs()) / m_outsize;};
        m_fploss = [this](const Vector& real, const Vector& output){return (output-real).unaryExpr([](double v){return (v*v-1)/abs(v*v-1) * v;}) / m_outsize;};
    }
    else if (m_loss_type == "CUSTOM")
    {
        m_floss = [this](const Vector& real, const Vector& output){return (0.5 * (output - real).array().pow(2))/ m_outsize;};
        m_fploss = [this](const Vector& real, const Vector& output){
            static double Nleft = 0, Nright = 0;
            double al = 0.8;
            //std::cout << Nleft << " " << Nright << " " << Nleft+Nright << std::endl;
            if (real[0]<0)
            {
                Vector val = (output-real).array() / m_outsize;
                if (val[0] < 0){ if(Nleft<0){val *= 1;}else{val *= -1;}; Nleft = al*Nleft-(1-al);};
                if (val[0] > 0){ if(Nleft>0){val *= 1;}else{val *= -1;}; Nleft = al*Nleft+(1-al);};
                return Nleft != 0 ? val : Vector::Zero(real.size());
            }
            else
            {
                Vector val = (output-real).array() / m_outsize;
                if (val[0] < 0){ if(Nright<0){val *= 1;}else{val *= -1;}; Nright = al*Nright-(1-al);};
                if (val[0] > 0){ if(Nright>0){val *= 1;}else{val *= -1;}; Nright = al*Nright+(1-al);};
                return Nright != 0 ? val : Vector::Zero(real.size());
            }
        };
    }
    else if (m_loss_type == "HEAVY")
    {
        double delta = 0.5, val = 190.;
        m_floss = [this, delta, val](const Vector& real, const Vector& output){auto v = (0.5 * (output - real).array().pow(2))/ m_outsize ;
            auto h = val * (output-real).unaryExpr([delta](double a){return abs(a) < delta ? 0 : abs(a) - delta;}).array();
            return v + h;
        };
        m_fploss = [this, delta, val](const Vector& real, const Vector& output){auto v = ((output - real).array())/ m_outsize ;
            auto h = val * (output - real).unaryExpr([delta](double a){return abs(a) < delta ? 0 :
                ( a > 0 ? 1. : -1.);}).array();
            return v + h;
        };
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
        double a = -200.;
        double c =  1.;
        double A = 1.;
        auto f =   [a, c, A](double x) {return A*abs(a-2.)/a * ( pow(pow(x/c, 2) / abs(a-2.) + 1., a/2.) - 1.);};
        auto df =  [a, c, A](double x) {return  A*x/(c*c) * ( pow(pow(x/c, 2) / abs(a-2.) + 1., a/2. - 1) );};
        m_floss =  [this, f](const Vector& real, const Vector& output) {return (output-real).unaryExpr(f) / m_outsize; };
        m_fploss = [this, df](const Vector& real, const Vector& output) {return (output-real).unaryExpr(df) / m_outsize; };
    }
    else if (m_loss_type == "LSC")
    {
        auto magn = [](double a) {return abs(a);};
        m_floss = [this, magn](const Vector& real, const Vector& output){return 0.5*((output-real).array().pow(2) +
                                                                                (output.unaryExpr(magn)-Vector::Constant(m_outsize, 1)).array().pow(2)) / m_outsize ;};
        auto sign = [](double a) {
            if ( a > 0)
                return +1.;
            else if ( a < 0)
                return -1.;
            else
                return 0.;
        };
        m_fploss = [this, sign, magn](const Vector& real, const Vector& output){return ((output-real).array() + 
                                            (output.unaryExpr(magn)-Vector::Constant(m_outsize, 1)).array()*output.array().unaryExpr(sign)) / m_outsize;};
    }
    else
        throw std::invalid_argument("this loss function isn't implemented");
}

void LayerDeque::set_adagrad_rate(double adagrad_rate)
{
    m_adagrad_rate = adagrad_rate;
}

void LayerDeque::set_dropout_rate(double dropout_rate)
{
    m_dropout_rate = dropout_rate;
}

void LayerDeque::set_regulization_rate(double regulization_rate)
{
    m_regulization_rate = regulization_rate;
}

void LayerDeque::set_viscosity_rate(double viscosity_rate)
{
    m_viscosity_rate = viscosity_rate;
}

void LayerDeque::set_step(const double step)
{
    m_step = step;
}

std::array<double, 3> LayerDeque::test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                        const std::vector<std::vector<double>>& weights) const
{
    if (input.size() != output.size())
        throw std::invalid_argument("input size is not equal to output size of testing data"); // TODO: make an global Exception static class

    if (weights.size() != output.size())
        throw std::invalid_argument("event weights size is not equal to output size of testing data"); // TODO: make an global Exception static class
    
    // Update parameters
    for (auto &layer: m_layers)
        layer->reset_layer(m_pars_map);

    double mean = 0.;
    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        std::vector<double> calc = calculate(input.at(idx));
        int calc_size = calc.size();
        const Vector Ycalc = Eigen::Map<const Vector, Eigen::Aligned>(calc.data(), calc_size);
        
        int out_size = output.at(idx).size();
        const Vector Yreal = Eigen::Map<const Vector, Eigen::Aligned>(output.at(idx).data(), out_size);

        int weights_size = output.at(idx).size();
        const Vector W = Eigen::Map<const Vector, Eigen::Aligned>(weights.at(idx).data(), weights_size);

        mean += (W.array() * m_floss(Yreal, Ycalc).array()).sum();
    }
    mean /= output.size();

    double stddev = 0.;
    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        std::vector<double> calc = calculate(input.at(idx));
        int calc_size = calc.size();
        const Vector Ycalc = Eigen::Map<const Vector, Eigen::Aligned>(calc.data(), calc_size);
        
        int out_size = output.at(idx).size();
        const Vector Yreal = Eigen::Map<const Vector, Eigen::Aligned>(output.at(idx).data(), out_size);

        int weights_size = output.at(idx).size();
        const Vector W = Eigen::Map<const Vector, Eigen::Aligned>(weights.at(idx).data(), weights_size);

        stddev += pow((W.array() * m_floss(Yreal, Ycalc).array()).sum() - mean, 2);
    }
    stddev = output.size() > 1? stddev / (output.size() - 1): 0;

    return {mean + get_regulization(), sqrt(stddev), mean};
}

void LayerDeque::train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                       const std::vector<std::vector<double>>& weights, unsigned int batch_size, unsigned int minibatch_size)
{
    if (input.size() != output.size())
        throw std::invalid_argument("input size is not equal to output size of training data"); // TODO: make an global Exception static class

    if (weights.size() != output.size())
        throw std::invalid_argument("event weights size is not equal to output size of training data"); // TODO: make an global Exception static class

    auto set_trainMode = [this] (bool flag)
    {
        for (auto &layer: m_layers)
        {
            layer->m_trainMode = flag;
            layer->reset_layer(m_pars_map);
        }
    };

    set_trainMode(true);

    // Calculate gradients and update weights
    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        if (m_layers.front()->size() != input.at(idx).size())
            throw std::invalid_argument("wrong input size"); // TODO: make an global Exception static class

        if (m_layers.back()->size() != output.at(idx).size())
            throw std::invalid_argument("wrong output size"); // TODO: make an global Exception static class
  
        // Calculate for one event a minibatch gradient 
        for (unsigned int jdx = 0; jdx < minibatch_size; ++jdx)
        {
            for (auto& layer: m_layers)
                layer->update();

            std::vector<std::pair<Matrix, Vector>> dL( std::move(get_gradient(input.at(idx), output.at(idx), weights.at(idx))) );

            for (unsigned idl = 0; idl < m_layers.size() - 1; ++idl)
            {
                //Calculate summary layer gradient with likelihood and regulization
                dL.at(idl).first *= 1./minibatch_size;
                dL.at(idl).second *= 1./minibatch_size;
                m_layers.at(idl)->add_gradient(dL.at(idl), batch_size);
            }
        }

        if ( (idx + 1) % batch_size == 0)
        {
            for (unsigned int idl = 0; idl < m_layers.size() - 1; ++idl)
            {
                m_layers.at(idl)->update_weights(m_step);
            }

            // is it necessary? it should be moved upper
            if (input.size() - idx < batch_size) // nothing to batch
                break;
        }
    }

    set_trainMode(false);
}

Vector LayerDeque::train_event(const std::vector<double>& input, const std::vector<double>& output)
{
    auto set_trainMode = [this] (bool flag)
    {
        for (auto &layer: m_layers)
        {
            layer->m_trainMode = flag;
            layer->reset_layer(m_pars_map);
        }
    };

    // Calculate gradients and update weights
    if (m_layers.front()->size() != input.size())
        throw std::invalid_argument("wrong input size"); // TODO: make an global Exception static class

    if (m_layers.back()->size() != output.size())
        throw std::invalid_argument("wrong output size"); // TODO: make an global Exception static class

    set_trainMode(true);
  
    // Calculate for one event a minibatch gradient 
    for (auto& layer: m_layers)
        layer->update();

    std::vector<double> weights(output.size(), 1.);

    std::vector<std::pair<Matrix, Vector>> dL( std::move(get_gradient(input, output, weights)) );

    for (unsigned idl = 0; idl < m_layers.size() - 1; ++idl)
    {
        //Calculate summary layer gradient with likelihood and regulization
        m_layers.at(idl)->add_gradient(dL.at(idl), 1);
    }

    for (unsigned int idl = 0; idl < m_layers.size() - 1; ++idl)
    {
        m_layers.at(idl)->update_weights(m_step);
    }

    set_trainMode(false);

    return dL.front().second * m_layers.front()->get_matrixW().transpose();
}

std::vector<std::pair<Matrix, Vector>> LayerDeque::get_gradient(const std::vector<double>& input, const std::vector<double>& output,
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

    // Xi = f(Zi) = f(sum(Wij*xj) + Bi), i - output neuron, j - input neurons
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

    Vector df = m_fploss(Y, *pX_layers.back()) + m_addition_gradient;
    Vector delta = (W.array() * df.array()) *
        m_layers.back()->calculateXp( *pZ_layers.back() ).array();

    if (m_alpha != 0)
    {
        Vector f = m_floss(Y, *pX_layers.back());

        m_ema *= m_alpha;
        m_ema += (1 - m_alpha) * f;

        Vector dx = ( ((2-m_alpha)*m_ema.array() - (1-m_alpha)*f.array())* m_ema.array().pow(2).inverse() - 1. );

        delta.array() *= dx.array();
    }


    for (int idx = m_layers.size()-2; idx > -1; --idx)
    {
        dL.emplace_back( std::pair<Matrix, Vector>((*pX_layers.at(idx)).transpose() * delta, delta));

        if (idx == 0)
            break;
            
        delta = m_layers.at(idx)->calculateXp( *pZ_layers.at(idx) ).array() *
 	    (delta * m_layers.at(idx)->get_matrixW().transpose()).array();
    }

    // Let's try to skip a gradient explosion
    //auto f = [](double el) {return std::isnan(el)? 0: el;};
    //for (auto& p: dL)
    //{
    //    p.first = p.first.unaryExpr(f);
    //    p.second = p.second.unaryExpr(f);
    //}

    std::reverse( std::begin(dL), std::end(dL) );
    return dL;
}

double LayerDeque::get_regulization() const
{
    if (m_regulization_rate == 0)
        return 0;

    double regulization = 0.;
    for (const auto& pLayer: m_layers)
        regulization += pLayer->get_regulization();
    return regulization / m_outsize;
}

const std::vector<std::vector<double>> LayerDeque::get_calculatedX(const std::vector<double>& input) const
{
    if (m_layers.front()->size() != input.size())
        throw std::invalid_argument("wrong input size, it is not equal to input layer size"); // TODO: make an global Exception static class

    std::vector<std::vector<double>> res;

    int size = input.size();
    Vector tmpZ( std::move(Eigen::Map<const Vector, Eigen::Aligned>(input.data(), size)) );
    Vector tmpX;
    for (const auto& pLayer: m_layers)
    {
        tmpX = pLayer->calculateX(tmpZ);
        res.push_back(std::vector<double>(tmpX.data(), tmpX.data() + tmpX.size()));
        tmpZ = pLayer->calculate(tmpZ);
    }

    return res;
}

const std::vector<std::vector<double>> LayerDeque::get_calculatedZ(const std::vector<double>& input) const
{
    if (m_layers.front()->size() != input.size())
    throw std::invalid_argument("wrong input size, it is not equal to input layer size"); // TODO: make an global Exception static class

    std::vector<std::vector<double>> res;

    int size = input.size();
    Vector tmpZ( std::move(Eigen::Map<const Vector, Eigen::Aligned>(input.data(), size)) );
    for (const auto& pLayer: m_layers)
    {
        tmpZ = pLayer->calculate(tmpZ);
        res.push_back(std::vector<double>(tmpZ.data(), tmpZ.data() + tmpZ.size()));
    }

    return res;
}