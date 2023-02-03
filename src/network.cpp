/**
 * @file network.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-09-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "../include/network.h"

static unsigned int MAX_INPUT_SIZE = 32;
static unsigned int MAX_OUTPUT_SIZE = 32;
static unsigned int MAX_LAYERS = 5;
static unsigned int MAX_LAYER_SIZE = 32;


Network::Exception::Exception(const char* message)
{
    m_message = std::string(message);
}

Network::Network():
    m_nepoch(0)
{
}

Network::~Network()
{
}

void Network::add_layers()
{
    m_layer_deque.add_layers<Layer>(m_topology);
}

void Network::add_in_transform(const std::string& type)
{
    if (type == "linear")
        m_in_transf.push_back(std::make_shared<LinearTransformation>());
}

void Network::add_in_transform(const std::string& type, const std::vector<double>& vars)
{
    if (type == "linear")
        m_in_transf.push_back(std::make_shared<LinearTransformation>(vars.front(), vars.back()));
}

void Network::add_out_transform(const std::string& type)
{
    if (type == "linear")
        m_out_transf.push_back(std::make_shared<LinearTransformation>());
}

void Network::add_out_transform(const std::string& type, const std::vector<double>& vars)
{
    if (type == "linear")
        m_out_transf.push_back(std::make_shared<LinearTransformation>(vars.front(), vars.back()));
}

bool Network::init_from_file(const std::string& in_file_name, const std::string& out_file_name="")
{
    std::ifstream fin(in_file_name);
    if (!fin.is_open())
    {
        std::cout << "Failed to open: " << in_file_name << std::endl;
        return false;
    }
    
    if (out_file_name != "")
        m_out_file_name = out_file_name;
    else
        m_out_file_name = "network.txt";

    std::string data;
    unsigned int deep;
    unsigned int layer_id = 0;
    while (fin >> data)
    {
        if (data == "numb_input")
        {
            fin >> m_numb_input;
            continue;
        }
        if (data == "numb_output")
        {
            fin >> m_numb_output;
            continue;
        }
        if (data == "deep")
        {
            fin >> deep;
            m_topology.reserve(deep);
            continue;
        }
        if (data == "topology" && deep > 0)
        {
            unsigned int size;
            for (unsigned int idx = 0; idx < deep; ++idx)
            {
                fin >> size;
                m_topology.emplace_back(size);
            }
            add_layers();
            continue;
        }
        if (data == "active_funcs" && deep > 0)
        {
            for (unsigned int i=0; i<deep; ++i)
            {
                fin >> data;
                m_active_funcs.push_back(data);
            }
            m_layer_deque.set_active_funcs(m_active_funcs);
            continue;
        }
        if (data == "layer")
        {
            if (layer_id + 1 >= deep)
                continue;
            m_layer_deque.read_layer(fin, layer_id);
            ++layer_id;
            continue;
        }
        if (data == "nepoch")
        {
            fin >> m_nepoch;
            continue;
        }
        if (data == "loss_func" )
        {
            fin >> data;
            m_loss = data;
            m_layer_deque.set_loss_func(data);
            continue;
        }
        if (data == "step" )
        {
            double step;
            fin >> step;
            m_layer_deque.set_step(step);
            continue;
        }
        if (data == "regulization_rate")
        {
            double rate;
            fin >> rate;
            m_layer_deque.set_regulization_rate(rate);
            continue;
        }
        if (data == "viscosity_rate")
        {
            double rate;
            fin >> rate;
            m_layer_deque.set_viscosity_rate(rate);
            continue;
	}
        if (data == "adagrad_rate")
        {
            double rate;
            fin >> rate;
            m_layer_deque.set_adagrad_rate(rate);
            continue;
	}
        if (data == "input_transforms" || data == "output_transforms")
        {
            unsigned int ntransforms = (data == "input_transforms")? m_numb_input: m_numb_output;
            for (unsigned int idx = 0; idx < ntransforms; ++idx)
            {
                std::string transform_type;
                fin >> transform_type;

                int nvars;
                fin >> nvars;
                if (nvars == 0)
                {
                    if (data == "input_transforms")
                        add_in_transform(transform_type);

                    if (data == "output_transforms")
                        add_out_transform(transform_type);
                    
                    continue;
                }
                else if (nvars > 0)
                {
                    std::vector<double> vars;
                    double var;
                    for (int ivar = 0; ivar < nvars; ++ivar)
                    {
                        fin >> var;
                        vars.push_back(var);
                    }

                    if (data == "input_transforms")
                        add_in_transform(transform_type, vars);

                    if (data == "output_transforms")
                        add_out_transform(transform_type, vars);
                    
                    continue;
                }
            }
            continue;
        }
    }
    if (m_in_transf.size() != m_numb_input)
        throw Exception("number of input transformations is not equal to number of input");
    
    if (m_out_transf.size() != m_numb_output)
        throw Exception("number of output transformations is not equal to number of output");

    // TODO: add more validations

    fin.close();
    print(std::cout);
    save();

    return true;
}

bool Network::create(unsigned int numb_input, unsigned int numb_output, const std::vector<unsigned int>& hidden_topology, const std::string& out_file_name)
{
    // TODO: move to assert?
    if (numb_input == 0 || numb_input > MAX_INPUT_SIZE)
        throw Exception((std::string("incorrect number of input variables! zero or more than") + std::to_string(MAX_INPUT_SIZE)).c_str());
    
    if (numb_output == 0 || numb_output > MAX_OUTPUT_SIZE)
        throw Exception((std::string("incorrect number of output variables! zero or more than") + std::to_string(MAX_OUTPUT_SIZE)).c_str());

    if (hidden_topology.size() == 0)
        throw Exception("network cannot have zero hidden layers!");

    if (hidden_topology.size() > MAX_LAYERS)
        throw Exception((std::string("network cannot have a lot hidden layers ") + std::to_string(MAX_LAYERS)).c_str());

    for (const auto& layer_size: hidden_topology)
    {
        if (layer_size == 0 || layer_size > MAX_LAYER_SIZE)
            throw Exception((std::string("incorrect layer size! one of the is zero or more than") + std::to_string(MAX_LAYER_SIZE)).c_str());
    }

    if (out_file_name.size() == 0 || !out_file_name.find(".txt"))
        throw Exception("incorrect output file name!");

    m_numb_input = numb_input;
    m_numb_output = numb_output;
    m_topology.reserve(hidden_topology.size() + 2); // plus input and iutput layer
    m_topology.emplace_back(numb_input);
    m_topology.insert(end(m_topology), begin(hidden_topology), end(hidden_topology));
    m_topology.emplace_back(numb_output);
    m_out_file_name = out_file_name;

    add_layers();

    m_active_funcs.push_back("linear");
    for (unsigned int idx = 0; idx < hidden_topology.size(); ++idx)
        m_active_funcs.push_back("sigmoid");
    m_active_funcs.push_back("linear");

    for (unsigned int idx = 0; idx < m_numb_input; ++idx)
        add_in_transform("linear");

    for (unsigned int idx = 0; idx < m_numb_output; ++idx)
        add_out_transform("linear");

    m_nepoch = 0;
    m_loss = "LS";
    m_layer_deque.set_loss_func("LS");

    m_layer_deque.generate_weights("Xavier");

    print(std::cout);
    save();
    return true;
}

void Network::print(std::ostream& os) const
{
    std::string result;
    os << "numb_input "     << m_numb_input << std::endl;
    os << "numb_output "    << m_numb_output << std::endl;
    os << "deep "           << m_topology.size() << std::endl;
    os << "nepoch "         << m_nepoch << std::endl;
    os << "loss_func "      << m_loss << std::endl;
    os << "step "           << m_layer_deque.get_step() << std::endl;
    os << "topology ";
    for (const auto& size: m_topology)
        os << size << " ";
    os << std::endl;

    os << "active_funcs ";
    for (const auto& func: m_active_funcs)
        os << func << " ";
    os << std::endl;

    os << "input_transforms" << std::endl;
    for(const auto& pTransform: m_in_transf)
    {
        os << "\t";
        pTransform->print(os);
    }

    os << "output_transforms" << std::endl;
    for(const auto& pTransform: m_out_transf)
    {
        os << "\t";
        pTransform->print(os);
    }


    m_layer_deque.print(os);
}

void Network::save() const
{
    std::ofstream fout(m_out_file_name.c_str());
    print(fout);
    fout.close();
}

std::vector<double> Network::get_result(const std::vector<double>& input) const
{    
    std::vector<double> transf_input(input.begin(), input.end());
    transform_input(transf_input);
    std::vector<double> transf_output = m_layer_deque.calculate(transf_input);
    reverse_transform_output(transf_output);
    return transf_output;
}

double Network::test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                     const std::vector<std::vector<double>>& weights) const
{
    const unsigned int input_size = input.size();
    const unsigned int output_size = output.size();
    const unsigned int weights_size = weights.size();
    if (input_size == 0 || output_size == 0)
        throw Exception("incorrect size of input/output! it is zero!");

    if (input_size != output_size)
        throw Exception("size of input and output are not equal");

    if (weights_size != output_size)
        throw Exception("size of event weights and output are not equal");
    
    return m_layer_deque.test(input, output, weights);
}

double Network::test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output) const
{
    std::vector<double> one_vec(m_numb_output, 1);
    std::vector<std::vector<double>> weights(output.size(), one_vec);

    return test(input, output, weights);
}

void Network::train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                    const std::vector<std::vector<double>>& weights, unsigned int batch_size, unsigned int minibatch_size, double split_mode)
{
    const unsigned int input_size = input.size();
    const unsigned int output_size = output.size();
    const unsigned int weights_size = weights.size();
    if (input_size == 0 || output_size == 0)
        throw Exception("incorrect size of input/output! it is zero!");
    
    if (input_size != output_size)
        throw Exception("size of input and output are not equal");

    if (weights_size != output_size)
        throw Exception("size of event weights and output are not equal");
    
    if (batch_size > input.size() || batch_size == 0)
        throw Exception("batch size is larger then input size or iz zero");
        
    // Set config of input and output normalization transformation
    std::for_each(input.begin(), input.end(), [this](const std::vector<double>& vars){
        for(unsigned int idx = 0; idx < m_in_transf.size(); ++idx)
            m_in_transf.at(idx)->check_limits(vars.at(idx));
    });

    std::for_each(output.begin(), output.end(), [this](const std::vector<double>& vars){
        for(unsigned int idx = 0; idx < m_out_transf.size(); ++idx)
            m_out_transf.at(idx)->check_limits(vars.at(idx));
    });

    std::for_each(m_in_transf.begin(), m_in_transf.end(), [](TransformationPtr& pTransf){pTransf->set_config();});
    std::for_each(m_out_transf.begin(), m_out_transf.end(), [](TransformationPtr& pTransf){pTransf->set_config();});

    // Normalize input vectors
    std::vector<std::vector<double>> transf_input(input.begin(), input.end());
    std::for_each(transf_input.begin(), transf_input.end(),
        [this](std::vector<double>& in_vars){transform_input(in_vars);
    });
    

    // Normalize output vectors
    std::vector<std::vector<double>> transf_output(output.begin(), output.end());
    std::for_each(transf_output.begin(), transf_output.end(),
        [this](std::vector<double>& out_vars){transform_output(out_vars);
    });

    // Normalize weights vector
    // DONOT implemented

    std::vector<std::vector<double>> train_input(transf_input.begin(), transf_input.begin() + int(split_mode * input_size));
    std::vector<std::vector<double>> train_output(transf_output.begin(), transf_output.begin() + int(split_mode * output_size));
    std::vector<std::vector<double>> train_weights(weights.begin(), weights.begin() + int(split_mode * input_size));
    
    std::vector<std::vector<double>> test_input(transf_input.begin() + int(split_mode * input_size), transf_input.end());
    std::vector<std::vector<double>> test_output(transf_output.begin() + int(split_mode * output_size), transf_output.end());
    std::vector<std::vector<double>> test_weights(weights.begin() + int(split_mode * input_size), weights.end());

    double epsilon_before = m_layer_deque.test(test_input, test_output, test_weights);
    m_layer_deque.train(train_input, train_output, train_weights, batch_size, minibatch_size);
    double epsilon_after = m_layer_deque.test(test_input, test_output, test_weights);

    double reduce = std::abs(epsilon_after / epsilon_before);

    /*// Fading descent
    double fading = std::exp(-reduce); // dependence on the previous result
    m_layer_deque.set_step(m_layer_deque.get_step() * reduce);
    */

    // Stochastic descent
    double period = 30.;
    double amp = .5;
    double step = amp * 0.5 * (1. + cos(std::abs(m_nepoch / period - int(m_nepoch / period / M_PI) * M_PI )));
    step = step == 0? amp: step;
    m_layer_deque.set_step(step);

    std::cout << "Nepoch: " << m_nepoch << " step: " << step << std::endl;
    std::cout << epsilon_before << " -> " << epsilon_after << " : " << reduce << std::endl;

    ++m_nepoch;
}

void Network::train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                    unsigned int batch_size, unsigned int minibatch_size, double split_mode)
{
    std::vector<double> one_vec(m_numb_output, 1);
    std::vector<std::vector<double>> weights(output.size(), one_vec);

    train(input, output, weights, batch_size, minibatch_size, split_mode);
}

void Network::transform_input(std::vector<double>& in_value) const
{
    if (in_value.size() != m_in_transf.size())
        throw Exception("size of input is not equal to transformations number");

    for (unsigned int idx = 0; idx < in_value.size(); ++idx)
        in_value.at(idx) = m_in_transf.at(idx)->transform(in_value.at(idx));
}

void Network::transform_output(std::vector<double>& out_value) const
{
    if (out_value.size() != m_out_transf.size())
        throw Exception("size of output is not equal to transformations number");

    for (unsigned int idx = 0; idx < out_value.size(); ++idx)
        out_value.at(idx) = m_out_transf.at(idx)->transform(out_value.at(idx));
}

void Network::reverse_transform_output(std::vector<double>& out_value) const
{
    if (out_value.size() != m_out_transf.size())
        throw Exception("size of normalized vector is not equal to transformations number");

    for (unsigned int idx = 0; idx < out_value.size(); ++idx)
        out_value.at(idx) = m_out_transf.at(idx)->reverse_transform(out_value.at(idx));
}

// TODO: Make an event weighting (+weight normalization), assembly of networks, also and randomazing of data order.
// TODO: Make a cost and test method. First with regulization and second without.

// Batch normalization? Normalization step that fixes the means and variances of each layer's inputs. Hence, every activation func should be normalized.
// It means that active func's variables can be changed by data. For example, squeeze and shift sigmoid to transform data form range [a, b] to [-1, +1]
// Actually, now it looks hard to realize. I should refactor whole code to good form.
