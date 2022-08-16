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
    std::cout << "Hello Network" << std::endl;
}

Network::~Network()
{
    m_topology.clear();
    m_input.clear();
    m_output.clear();
    m_layers.clear();
    std::cout << "Buy Network" << std::endl;
}

void Network::init_from_file(const std::string& in_file_name, const std::string& out_file_name="")
{
    std::cout << in_file_name << " " << out_file_name << std::endl;
    return;
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

    net->m_layers.add_layers(net->m_topology);
    net->m_layers.generate_weights("Xavier");

    net->print(std::cout);
    return net;
}

void Network::print(std::ostream& os) const
{
    std::string result;
    os << "numb_input " + std::to_string(m_numb_input) + "\n";
    os << "numb_output " + std::to_string(m_numb_output) + "\n";
    os << "topology: ";
    for (const auto& size: m_topology)
        std::cout << size << " -> ";
    std::cout << "\n";

    m_layers.print(os);
}

std::vector<double> Network::get_result(const std::vector<double>& input) const{
    return m_layers.calculate(input);
}