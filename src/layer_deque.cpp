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

    if (m_useKDE)
    {
        m_kde_local = std::make_unique<KDE>(1);
        m_kde_global = std::make_unique<KDE>(0);
        //m_kde->set_verbose();
    }

    if (m_useBinningKDE)
    {
        for (int i = 0; i < m_num; i++)
            m_kdes.push_back(std::make_unique<KDE>(0));
    }

    if (m_useKS)
    {
        m_ks = std::make_unique<KStest>(0.1, [](double x) {
            double a = sqrt(2) * 0.2;
            return -0.141047*a*pow(2.71828, -pow((1-x)/a, 2)) + 0.141047*a*pow(2.71828, -pow((1+x)/a, 2))
                + (0.25*x - 0.25) * std::erf((1-x)/a) + (0.25*x + 0.25) * std::erf((1+x)/a) + 0.5;
        });
        //m_ks->set_verbose();
    }

    if (m_neighbourSum)
    {
        m_sortedPlaces.clear();
        m_sortedIndeces.clear();
        std::ifstream istrm(m_savedNeighbours, std::ios::binary);
        if (!istrm.is_open())
            std::cout << "Neighbours::Failed to open file, it will be stored soon!" << std::endl;
        else
        {
            int place, ind;
            while (istrm >> place >> ind)
            {
                m_sortedPlaces.push_back(place);
                m_sortedIndeces.push_back(ind);
            }
        }

        istrm.close();
    }
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

void LayerDeque::prepare_batch(const std::vector<std::vector<double>>& input,
                               const std::vector<std::vector<double>>& output,
                               unsigned int idx, unsigned int batch_size)
{
    double val = 0;
    if (m_useKDE)
    {
        if (idx % batch_size == 0)
        {
            for (auto &layer: m_layers)
                layer->m_trainMode = false;

            // Fill array
            std::vector<double> data, reco_local, reco_global;
            for (unsigned int i = idx; i < idx + batch_size; i++)
            {
                data.push_back(output.at(i).at(0));
                reco_local.push_back(calculate(input.at(i)).at(0) - output.at(i).at(0));
                reco_global.push_back(calculate(input.at(i)).at(0));
            }
    
            //m_kde_local->recalculate_exclusive(reco_local);
            //m_kde_global->recalculate_exclusive(reco_global);
            //m_kde_global->fast_recalculate(reco_global);
            m_kde_global->recalculate_inclusive(data, reco_global);
    
            for (auto &layer: m_layers)
                layer->m_trainMode = true;
        }

        //val += 1e1 * m_kde_local->get_gradient( idx % batch_size);
        val += 1e1 * m_kde_global->get_gradient( idx % batch_size);
    }

    if (m_useBinningKDE)
    {
        if (idx % batch_size == 0)
        {
            m_ids.clear();
	        for (int i = 0; i < m_num; i++)
            {
                m_kde_left[i] = 0;
                m_kde_right[i] = 0;
	            m_nums_kde_left[i] = 0;
	            m_nums_kde_right[i] = 0;
	        }

            for (auto &layer: m_layers)
                layer->m_trainMode = false;

            // Fill arrays
            std::vector<std::vector<double>> reco_local(m_num);
            for (unsigned int i = idx; i < idx + batch_size; i++)
            {
                double simen = output.at(i).at(0);
                double reco = calculate(input.at(i)).at(0);
                int mid = ((simen + 1) * 0.5 * m_num);
                if (mid == m_num)
                    mid--;

                m_ids.push_back(reco_local.at(mid).size());
                reco_local.at(mid).push_back(reco - simen);
                
                if (reco - simen < 0)
                {
                    m_kde_left[mid] += pow(reco - simen, 2);
                    m_nums_kde_left[mid]++;
                }
                else
                {
                    m_kde_right[mid] += pow(reco - simen, 2);
                    m_nums_kde_right[mid]++;
                }
            }
    
            for (int i = 0; i < m_num; i++)
            {
                m_kde_left[i] = sqrt(m_kde_left[i] / m_nums_kde_left[i]);
                m_kde_right[i] = sqrt(m_kde_right[i] / m_nums_kde_right[i]);
            }

            for (int i = 0; i < m_num; i++)
            {
                m_kdes.at(i)->set_parameters(m_kde_left[i], m_kde_right[i]);
                m_kdes.at(i)->recalculate_exclusive(reco_local.at(i));
            }

            for (auto &layer: m_layers)
                layer->m_trainMode = true;
        }

        double simen = output.at(idx).at(0);

        for (auto &layer: m_layers)
            layer->m_trainMode = false;

        double reco = calculate(input.at(idx)).at(0);

        for (auto &layer: m_layers)
            layer->m_trainMode = true;

        int mid = ((simen + 1) * 0.5 * m_num);
        if (mid == m_num)
            mid--;

        val += 5e-1 * (
                1e1 * m_kdes.at(mid)->get_gradient(m_ids.at(idx % batch_size)) + 
                2 * (m_kde_left[mid] * 0.7 - m_kde_right[mid]) * 
                (reco - simen) / (reco - simen < 0 ? m_kde_left[mid]/0.7 : -m_kde_right[mid])
        );
    }

    if (m_useBinningZeroMean)
    {
        if (idx % batch_size == 0)
        {
	        for (auto &layer: m_layers)
	            layer->m_trainMode = false;

	        for (int i = 0; i < m_num; i++)
            {
	            m_means[i] = 0;
                m_std_left[i] = 0;
                m_std_right[i] = 0;
	            m_nums[i] = 0;
	            m_nums_left[i] = 0;
	            m_nums_right[i] = 0;
	        }

	        for (unsigned int i = idx; i < idx + batch_size; i++)
	        {
	            double simen = output.at(i).at(0);
                int mid = ((simen + 1) * 0.5 * m_num);
                if (mid == m_num)
                    mid--;
        
                if (mid > m_num || mid < 0)
                    throw std::invalid_argument("invalid num of bin");

                double calc = calculate(input.at(i)).at(0);
                if (calc < simen)
                {
                    m_std_left[mid] += pow(calc - simen, 2);
                    m_nums_left[mid]++;
                }
                else
                {
                    m_std_right[mid] += pow(calc - simen, 2);
                    m_nums_right[mid]++;
                }

	        }

	        for (int i = 0; i < m_num; i++)
            {
	        	m_std_left[i] = sqrt(m_std_left[i]/m_nums_left[i]);
	        	m_std_right[i] = sqrt(m_std_right[i]/m_nums_right[i]);
            }

	        for (unsigned int i = idx; i < idx + batch_size; i++)
	        {
	            double simen = output.at(i).at(0);
                int mid = ((simen + 1) * 0.5 * m_num);
                if (mid == m_num)
                    mid--;
        
                if (mid > m_num || mid < 0)
                    throw std::invalid_argument("invalid num of bin");

                double calc = calculate(input.at(i)).at(0);
                m_means[mid] += (calc - simen) / (calc < simen ? m_std_left[mid] : m_std_right[mid]);
                m_nums[mid]++;
	        }

	        for (int i = 0; i < m_num; i++)
	        	m_means[i] /= m_nums[i];

	        for (auto &layer: m_layers)
	            layer->m_trainMode = true;
        }

        int mid = ((output.at(idx).at(0) + 1) * 0.5 * m_num);
        if (mid == m_num)
            mid--;

        if (mid > m_num || mid < 0)
            throw std::invalid_argument("invalid num of bin");

	    for (auto &layer: m_layers)
	        layer->m_trainMode = false;

        double calcus = calculate(input.at(idx)).at(0);

	    for (auto &layer: m_layers)
	        layer->m_trainMode = true;

	    val += 1e1 * 2 * m_means[mid] / (calcus < output.at(idx).at(0) ? m_std_left[mid] : m_std_right[mid]) +
            1e2 * (m_std_left[mid]*0.72 - m_std_right[mid]) / (calcus < output.at(idx).at(0) ? m_std_left[mid] / 0.72 : -m_std_right[mid]) *
            (calcus - output.at(idx).at(0));
    }

    if (m_useKS)
    {
        if (idx % batch_size == 0)
        {
            for (auto &layer: m_layers)
                layer->m_trainMode = false;

            // Fill array
            std::vector<double> reco;
            for (unsigned int i = idx; i < idx + batch_size; i++)
            {
                //reco.push_back(calculate(input.at(i)).at(0) - output.at(i).at(0));
                reco.push_back(calculate(input.at(i)).at(0));
            }
    
            m_ks->calculateKS(reco);
    
            for (auto &layer: m_layers)
                layer->m_trainMode = true;
        }

        val += 1e3 * m_ks->get_gradient( idx % batch_size);
    }

    if (m_useZeroSlope)
    {
        if (idx % batch_size == 0)
        {
            m_ls_data.clear();
            for (auto &layer: m_layers)
                layer->m_trainMode = false;

            std::vector<double> delta, sim, rec;
            for (unsigned int i = idx; i < idx + batch_size; i++)
            {
                double r = calculate(input.at(i)).at(0);
                double s = output.at(i).at(0);
                delta.push_back(r-s);
                rec.push_back(r);
                sim.push_back(s);
            }

            for (auto &layer: m_layers)
                layer->m_trainMode = true;

            m_ls_data.push_back(LinearLS::calculateKB(sim, delta));

            for (int ipar = 0; ipar < input.at(0).size(); ipar++)
            {
                std::vector<double> x;
                for (unsigned int i = idx; i < idx + batch_size; i++)
                    x.push_back(input.at(i).at(ipar));

                m_ls_data.push_back(LinearLS::calculateKB(x, delta));
            }

            for (int ipar = 0; ipar < input.at(0).size(); ipar++)
            {
                std::vector<double> x;
                for (unsigned int i = idx; i < idx + batch_size; i++)
                    x.push_back(input.at(i).at(ipar));

                m_ls_data.push_back(LinearLS::calculateKB(x, rec));
            }
        }

        int n = output.size();
        double lambda = 1e2;
        double ksim = m_ls_data.at(0).at(0);
        double bsim = m_ls_data.at(0).at(1);
        double sumXsim = m_ls_data.at(0).at(2);
        double sumX2sim = m_ls_data.at(0).at(3);
        double grad_k = ksim * (n * output.at(idx).at(0) - sumXsim) / (n * sumX2sim - sumXsim * sumXsim) * n;
        double grad_b = bsim * (1 - grad_k * sumXsim) / n * n;
        val += lambda*(grad_k + grad_b);

        //for (int idata = 1; idata < m_ls_data.size(); idata++)
        //{
        //    //if (idata == 2 || (idata > 5 && idata < 8))
        //    if ((idata > 5 && idata < 8))
        //        continue;

        //    double k = m_ls_data.at(idata).at(0);
        //    double b = m_ls_data.at(idata).at(1);
        //    double sumX = m_ls_data.at(idata).at(2);
        //    double sumX2 = m_ls_data.at(idata).at(3);

        //    double gr_k = k * (n * input.at(idx).at((idata-1) % 5) - sumX) / (n * sumX2 - sumX * sumX) * n;
        //    double gr_b = 0;//b * (1 - gr_k * sumX) / n * n;
        //    val += lambda * (gr_k + gr_b);
        //}
    }

    if (m_neighbourSum)
    {
        auto it = m_sortedIndeces.begin() + m_sortedPlaces.at(idx);
        if (*it != idx)
        {
            std::cout << "Neighbours::Error! Wrong indeces!" << std::endl;
        }

        auto istart = std::distance(m_sortedIndeces.begin(), it) > m_kNeighbours ? it - m_kNeighbours : m_sortedIndeces.begin();
        auto iend = std::distance(it, m_sortedIndeces.end()) > m_kNeighbours ? it + m_kNeighbours : m_sortedIndeces.end();
        
        for (auto &layer: m_layers)
            layer->m_trainMode = false;

        double sum_rec = 0, sum_sim = 0;
        for (auto jt = istart; jt != iend; ++jt)
        {
            int id = *jt;
            double r = calculate(input.at(id)).at(0);
            double s = output.at(id).at(0);
            sum_rec += r;
            sum_sim += s;
        }

        val += 1 * (sum_rec - sum_sim) / std::distance(istart, iend);
	    m_lastResByNeighbours += (sum_rec - sum_sim) / std::distance(istart, iend);
	    m_sigmaByNeighbours += pow((sum_rec - sum_sim) / std::distance(istart, iend), 2);

        for (auto &layer: m_layers)
            layer->m_trainMode = true;
    }

    if (m_correlation)
    {
        if (idx % batch_size == 0)
        {
            for (auto &layer: m_layers)
                layer->m_trainMode = false;

            m_corr = 0;
            m_s_mean = 0;
            m_r_sigma = 0;
            m_s_sigma = 0;

            double r_mean = 0, rs_mean = 0;
            for (unsigned int i = idx; i < idx + batch_size; i++)
            {
                double r = calculate(input.at(i)).at(0);
                double s = output.at(i).at(0);
                r_mean += r - s;
                m_s_mean += s;
                rs_mean += (r-s)*s;
                m_r_sigma += (r-s)*(r-s);
                m_s_sigma += s*s;
            }

            r_mean /= batch_size;
            m_s_mean /= batch_size;
            rs_mean /= batch_size;

            m_r_sigma /= batch_size;
            m_r_sigma -= pow(r_mean, 2);
            m_r_sigma = sqrt(m_r_sigma);

            m_s_sigma /= batch_size;
            m_s_sigma -= pow(m_s_mean, 2);
            m_s_sigma = sqrt(m_s_sigma);

            m_corr = (rs_mean - r_mean * m_s_mean) / (m_r_sigma * m_s_sigma);

            for (auto &layer: m_layers)
                layer->m_trainMode = true;
        }

        val += 1e3 * (m_corr) * (output.at(idx).at(0) - m_s_mean) / (m_r_sigma * m_s_sigma);
    }

    if (m_neigh)
    {
        for (auto &layer: m_layers)
            layer->m_trainMode = false;

        std::multiset<double> distances;
        for (unsigned int i = (idx / batch_size) * batch_size; i < (idx / batch_size + 1) * batch_size; i++)
        {
            // Search neighbours of current event inside batch
            distances.insert(abs(output.at(idx).at(0) - output.at(i).at(0)));
        }

        m_sum = 0;
        int count = 0; 
        double max_dist = *(std::next(distances.begin(), m_kNeigh));
        for (unsigned int i = (idx / batch_size) * batch_size; i < (idx / batch_size + 1) * batch_size; i++)
        {
            if (abs(output.at(idx).at(0) - output.at(i).at(0)) > max_dist)
                continue;
            
            m_sum += calculate(input.at(i)).at(0) - output.at(i).at(0);
            count++;
        }

        m_sum /= count;

        for (auto &layer: m_layers)
            layer->m_trainMode = true;

        val += 1e3 * m_sum;
    }

    set_addition_gradient(Vector::Constant(1, m_outsize, val));
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
    else if (m_loss_type == "HUBER")
    {
        double delta = 0.05;
        auto huber = [delta](double a) {return abs(a) < delta ?  0.5*pow(a, 2) : delta * (abs(a) - delta);};
        m_floss = [this, huber](const Vector& real, const Vector& output){return (output-real).unaryExpr(huber)/ m_outsize;};
        auto dhuber = [delta](double a) {return abs(a) < delta ?  a : delta * abs(a) / a;};
        m_fploss = [this, dhuber](const Vector& real, const Vector& output){return  (output-real).unaryExpr(dhuber) / m_outsize;};
    }
    else if (m_loss_type == "SMAE")
    {
        auto smae = [](double a) {return a * tanh(0.5*a);};
        m_floss = [this, smae](const Vector& real, const Vector& output){return (output-real).unaryExpr(smae)/ m_outsize;};
        auto dsmae = [](double a) {return tanh(0.5*a) + 0.5*a*1./pow(cosh(0.5*a), 2);};
        m_fploss = [this, dsmae](const Vector& real, const Vector& output){return  (output-real).unaryExpr(dsmae) / m_outsize;};
    }
    else if (m_loss_type == "LETSGO")
    {
        const double amp = 1.;
        const double alpha = 0.6;
        m_floss = [this, amp, alpha](const Vector& real, const Vector& output){
            auto smae = [](double a) {return a * tanh(0.5*a);};
            static Vector ema = Vector::Zero(real.size());
            ema = alpha * ema.array() + (1-alpha) * (output-real).array()*real.array();
            return ( (output - real).array().unaryExpr(smae) + amp * ema.array().abs() ) / m_outsize;
        };
        m_fploss = [this, amp, alpha](const Vector& real, const Vector& output){
            auto dsmae = [](double a) {return tanh(0.5*a) + 0.5*a*1./pow(cosh(0.5*a), 2);};
            static Vector ema = Vector::Zero(real.size());
            ema = alpha * ema.array() + (1-alpha) * (output-real).array() * real.array();
            Vector sign = ema.array().unaryExpr([](double v){return v > 0? 1.: -1.;});
            //std::cout << ema_xy - ema_x*ema_y << " " << ema_xy << " " << ema_x << " " << ema_y << std::endl;
            Vector df = ( (output - real).array().unaryExpr(dsmae) + amp * sign.array() * (1-alpha) * real.array() ) / m_outsize;
            return df;
        };
    }
    else if (m_loss_type == "MSLEL")
    {
        double alpha = 500;
        double epsilon = 1e-9;
        double scale = 0.8;
        m_floss = [this, alpha, epsilon, scale](const Vector& real, const Vector& output){
            auto square = [](double a) {return 0.5*pow(a, 2);};
            auto loge = [epsilon, scale](double x, double y) {
                                                    //return 0.5*pow(log(pow(1 - scale*pow(x-y, 2), 2)), 2);
                                                    if (scale*pow(x-y, 2) < 1)
                                                        return 0.5*pow(log(1 - scale*pow(x-y, 2)), 2);
                                                    else
                                                        return 0.5*pow(2*log(epsilon), 2);
            };
            return ( (output - real).array().unaryExpr(square) +
                      alpha * output.array().binaryExpr(real.array(), loge) ) / m_outsize;
        };
        m_fploss = [this, alpha, epsilon, scale](const Vector& real, const Vector& output){
            auto loge = [epsilon, scale](double x, double y) {
                                                    if (scale*pow(x-y, 2) < 1)
                                                        return 0.5*pow(log(1 - scale*pow(x-y, 2)), 2);
                                                    else
                                                        return 0.5*pow(2*log(epsilon), 2);
            };

            auto dloge = [epsilon, scale](double x, double y) {
                                                    if (scale*pow(x-y, 2) < 1)
                                                        return -log(1 - scale*pow(x-y, 2)) / (1 - scale*pow(x-y, 2)) * 2 * scale * (x-y);
                                                    else
                                                        return -2*log(epsilon) / (epsilon) * 2 * scale * (x-y);
            };
            return ( (output - real).array() +
                      alpha * output.array().binaryExpr(real.array(), dloge) ) / m_outsize;
        };
    }
    else if (m_loss_type == "LOGX")
    {
        m_floss = [this](const Vector& real, const Vector& output){
            auto loge = [](double x, double y) {
                                                return log(1 + abs(x - y));
            };
            return ( output.array().binaryExpr(real.array(), loge) ) / m_outsize;
        };
        m_fploss = [this](const Vector& real, const Vector& output){
            auto loge = [](double x, double y) {
                                                return (x - y > 0 ? 1 : -1) / (1 + abs(x - y));
            };
            return ( output.array().binaryExpr(real.array(), loge) ) / m_outsize;
        };

    }
    else if (m_loss_type == "MS")
    {
        m_floss = [this](const Vector& real, const Vector& output){return 0.25 * (output.array().pow(2) - real.array().pow(2) - 0.04).pow(2)/ m_outsize;};
        m_fploss = [this](const Vector& real, const Vector& output){return (output.array().pow(2)-real.array().pow(2) -0.04) * output.array() / m_outsize;};
    }
    else if (m_loss_type == "NSK")
    {
        m_floss = [this](const Vector& real, const Vector& output){
            auto nsk = [](double x, double y) {
                double eta = 0.13;
                double sigma = 0.2;//736;// + 0.0038 * y;
                double v = 1 - eta * (y - x) / sigma;
                if (v <= 0 )
                    return 0.;

                return 0.5 * pow(log(v), 2);
            };

            return output.array().binaryExpr(real.array(), nsk) / m_outsize;
        };

        m_fploss = [this](const Vector& real, const Vector& output){
            auto dnsk = [](double x, double y)
            {
                double eta = 0.13;
                double sigma = 0.2;//736;// + 0.0038 * y;
                double v = 1 - eta * (y - x) / sigma;
                if (v <= 0 )
                    return 0.;

                return log(v) / v * eta / sigma;
            };
 
            return output.array().binaryExpr(real.array(), dnsk) / m_outsize;
        };
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

    return {mean + get_regulization()/output.size(), sqrt(stddev), mean};
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

    if (m_neighbourSum && m_sortedIndeces.empty() && m_sortedPlaces.empty())
    {
        std::cout << "Start sort" << std::endl;
        std::multimap<double, int> sortedOutput;
        clock_t start, end;
        start = clock();

        m_sortedIndeces.clear();
        m_sortedPlaces.clear();

        for (unsigned int idx = 0; idx < output.size(); ++idx)
        {
            sortedOutput.emplace_hint(std::next(sortedOutput.begin(), (sortedOutput.size() * (output.at(idx).at(0) + 1) / 2.)),
                                      std::pair{output.at(idx).at(0), idx});
            if (idx % 1000 == 0)
                std::cout << sortedOutput.size() << "/" << output.size() << std::endl;
        }

        for (auto it = sortedOutput.begin(); it != sortedOutput.end(); ++it)
            m_sortedIndeces.push_back(it->second);

        double epsilon = 1e-4;
        for (unsigned int idx = 0; idx < output.size(); ++idx)
        {
            auto it = sortedOutput.lower_bound(output.at(idx).at(0) - epsilon);
            while (it->second != idx)
                ++it;

            m_sortedPlaces.push_back(std::distance(sortedOutput.begin(), it));

            if (idx % 1000 == 0)
                std::cout << m_sortedPlaces.size() << "/" << output.size() << std::endl;
        }

        std::ofstream ostrm(m_savedNeighbours, std::ios::binary);
        for (int id = 0; id < m_sortedPlaces.size(); ++id)
            ostrm << m_sortedPlaces.at(id) << " \t " << m_sortedIndeces.at(id) << std::endl;

        ostrm.close();        

        end = clock();
        std::cout << "Finish sort: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;
    }

    // Calculate gradients and update weights
    int M = input.size() / batch_size;
    for (unsigned int idx = 0; idx < input.size(); ++idx)
    {
        if (m_layers.front()->size() != input.at(idx).size())
            throw std::invalid_argument("wrong input size"); // TODO: make an global Exception static class

        if (m_layers.back()->size() != output.at(idx).size())
            throw std::invalid_argument("wrong output size"); // TODO: make an global Exception static class

        prepare_batch(input, output, idx, batch_size);
  
        // Calculate for one event a minibatch gradient 
        for (unsigned int jdx = 0; jdx < minibatch_size; ++jdx)
        {
            for (auto& layer: m_layers)
                layer->update();

            auto out = output.at(idx);
            //out.at(0) += m_generator.generate();
            //if (out.at(0) < -3)
            //    out.at(0) = -3;

            std::vector<std::pair<Matrix, Vector>> dL( std::move(get_gradient(input.at(idx), out, weights.at(idx))) );

            // Make parallel
            for (unsigned idl = 0; idl < m_layers.size() - 1; ++idl)
            {
                //Calculate summary layer gradient with likelihood and regulization
                dL.at(idl).first *= M;
                dL.at(idl).second *= M;
                m_layers.at(idl)->add_gradient(dL.at(idl), batch_size * minibatch_size * M);
            }
        }

        if ( (idx + 1) % batch_size == 0)
        {
            // Make parallel
            for (unsigned int idl = 0; idl < m_layers.size() - 1; ++idl)
            {
                m_layers.at(idl)->update_weights(m_step);
            }

            // is it necessary? it should be moved upper
            if (input.size() - idx < batch_size) // nothing to batch
                break;
        }
    }
    
    if (m_useKDE)
        std::cout << "KL: " << m_kde_global->get_kl() << " -> grad: " << m_kde_global->get_dkl() << std::endl;

    if (m_useBinningKDE)
    {
        for (int i = 0; i < m_num; i++)
        {
            std::cout << i+1 << ") KL: " << m_kdes.at(i)->get_kl() << " -> grad: " << m_kdes.at(i)->get_dkl();
            std::cout << " [" << m_kde_left[i] << ", " << m_kde_right[i] << "]" << std::endl;
        }
    }

    if (m_useKS)
        std::cout << "KS: " << m_ks->get_sup() << " x0: " << m_ks->get_x0() << std::endl;

    if (m_neighbourSum)
    {
        std::cout << "Neighbours: " << m_lastResByNeighbours << std::endl;
        std::cout << "sigma Neighbours: " << m_sigmaByNeighbours << std::endl;
	    m_lastResByNeighbours = 0;
	    m_sigmaByNeighbours = 0;
    }

    if (m_correlation)
        std::cout << "Correlation: " << m_corr << " sim mean:" << m_s_mean << " sim sigma: " << m_s_sigma << std::endl;

    if (m_neigh)
        std::cout << "Last sum: " << m_sum << std::endl;

    if (m_useZeroSlope)
    {
        std::cout << "simen \t k: " << m_ls_data.at(0).at(0) << ", b " << m_ls_data.at(0).at(1) << std::endl;
        std::cout << "lxe \t k: " << m_ls_data.at(1).at(0) << ", b " << m_ls_data.at(1).at(1) << std::endl;
        std::cout << "csi \t k: " << m_ls_data.at(2).at(0) << ", b " << m_ls_data.at(2).at(1) << std::endl;
        std::cout << "rho \t k: " << m_ls_data.at(3).at(0) << ", b " << m_ls_data.at(3).at(1) << std::endl;
        std::cout << "theta \t k: " << m_ls_data.at(4).at(0) << ", b " << m_ls_data.at(4).at(1) << std::endl;
        std::cout << "phi \t k: " << m_ls_data.at(5).at(0) << ", b " << m_ls_data.at(5).at(1) << std::endl;
        std::cout << "rho2 \t k: " << m_ls_data.at(8).at(0) << ", b " << m_ls_data.at(8).at(1) << std::endl;
        std::cout << "theta2 \t k: " << m_ls_data.at(9).at(0) << ", b " << m_ls_data.at(9).at(1) << std::endl;
        std::cout << "phi2 \t k: " << m_ls_data.at(10).at(0) << ", b " << m_ls_data.at(10).at(1) << std::endl;
    }

    if (m_useBinningZeroMean)
    {
        for (int i = 0; i < m_num; i++)
        {
            std::cout << m_means[i] << " [" << m_std_left[i] << ", " << m_std_right[i] << "]: ";
            std::cout << m_std_right[i]/m_std_left[i] << std::endl;
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
