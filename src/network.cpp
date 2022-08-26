#include "network.h"

static unsigned int MAX_INPUT_SIZE = 32;
static unsigned int MAX_OUTPUT_SIZE = 32;
static unsigned int MAX_LAYERS = 5;
static unsigned int MAX_LAYER_SIZE = 32;


Network::Exception::Exception(const char* message)
{
    this->m_message = std::string(message);
}

Network::Network()
{
}

Network::~Network()
{
}

Network* Network::init_from_file(const std::string& in_file_name, const std::string& out_file_name="")
{
    std::ifstream fin(in_file_name);
    if (!fin.is_open())
    {
        std::cout << "Failed to open: " << in_file_name << std::endl;
        return nullptr;
    }
    
    Network* net = new Network();

    if (out_file_name != "")
        net->m_out_file_name = out_file_name;
    else
        net->m_out_file_name = "network.txt";

    std::string data;
    std::vector<std::vector<double>> matrices;
    std::vector<std::vector<double>> vectors;
    while (fin >> data)
    {
        if (data == "numb_input")
        {
            fin >> net->m_numb_input;
            continue;
        }
        if (data == "numb_output")
        {
            fin >> net->m_numb_output;
            continue;
        }
        unsigned int deep;
        if (data == "deep")
        {
            fin >> deep;
            net->m_topology.reserve(deep);
            continue;
        }
        if (data == "topology" && deep > 0)
        {
            unsigned int size;
            for (unsigned int idx = 0; idx < deep; ++idx)
            {
                fin >> size;
                net->m_topology.emplace_back(size);
            }
            net->m_layer_deque.add_layers(net->m_topology);
            continue;
        }
        if (data == "active_funcs" && deep > 0)
        {
            for (unsigned int i=0; i<deep; ++i)
            {
                fin >> data;
                net->m_active_funcs.push_back(data);
            }
            net->m_layer_deque.set_active_funcs(net->m_active_funcs);
            continue;
        }
        if (data == "[")
        {
            std::vector<double> weights;
            while (true)
            {
                fin >> data;
                if (data == "]")
                    break;
                double weight = std::stod(data);
                weights.push_back(weight);
            }
            matrices.push_back(weights);
            continue;
        }
        if (data == "{")
        {
            std::vector<double> biases;
            while (true)
            {
                fin >> data;
                if (data == "}")
                    break;
                double bias = std::stod(data);
                biases.push_back(bias);
            }
            vectors.push_back(biases);
            continue;
        }
        if (data == "nepoch")
        {
            fin >> net->m_nepoch;
            continue;
        }
        if (data == "loss_func" )
        {
            fin >> data;
            net->m_loss = data;
            net->m_layer_deque.set_loss_func(data);
            continue;
        }
    }
    fin.close();
    net->m_layer_deque.set_layers(matrices, vectors);
    net->print(std::cout);
    std::ofstream fout(net->m_out_file_name.c_str());
    net->print(fout);
    fout.close();

    return net;
}

Network* Network::create(unsigned int numb_input, unsigned int numb_output, const std::vector<unsigned int>& hidden_topology, const std::string& out_file_name)
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

    Network* net = new Network();
    net->m_numb_input = numb_input;
    net->m_numb_output = numb_output;
    net->m_topology.reserve(hidden_topology.size() + 2); // plus input and iutput layer
    net->m_topology.emplace_back(numb_input);
    net->m_topology.insert(end(net->m_topology), begin(hidden_topology), end(hidden_topology));
    net->m_topology.emplace_back(numb_output);
    net->m_out_file_name = out_file_name;

    net->m_layer_deque.add_layers(net->m_topology);

    net->m_active_funcs.push_back("linear");
    for (unsigned int idx=0; idx<hidden_topology.size(); ++idx)
        net->m_active_funcs.push_back("sigmoid");
    net->m_active_funcs.push_back("linear");

    net->m_nepoch = 0;
    net->m_loss = "LS";
    net->m_layer_deque.set_loss_func("LS");

    net->m_layer_deque.generate_weights("Xavier");

    net->print(std::cout);
    std::ofstream fout(net->m_out_file_name.c_str());
    net->print(fout);
    fout.close();
    return net;
}

void Network::print(std::ostream& os) const
{
    std::string result;
    os << "numb_input "     << m_numb_input << std::endl;
    os << "numb_output "    << m_numb_output << std::endl;
    os << "deep "           << m_topology.size() << std::endl;
    os << "nepoch "         << m_nepoch << std::endl;
    os << "loss_func "      << m_loss << std::endl;
    os << "topology ";
    for (const auto& size: m_topology)
        os << size << " ";
    os << std::endl;

    os << "active_funcs ";
    for (const auto& func: m_active_funcs)
        os << func << " ";
    os << std::endl;

    m_layer_deque.print(os);
}

void Network::save() const
{
    std::ofstream fout(m_out_file_name.c_str());
    print(fout);
    fout.close();
}

std::vector<double> Network::get_result(const std::vector<double>& input) const{
    return m_layer_deque.calculate(input);
}

void Network::train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output, unsigned int batch_size)
{
    if (batch_size == 0)
        batch_size = 1;
    m_layer_deque.train(input, output, batch_size);
}

// TODO: Make a normalization of input and iutput data, and radnomazing of data order, also add testing