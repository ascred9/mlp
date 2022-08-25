#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>

#include "layer.h"


class Network
{
private:
    unsigned int m_numb_input;
    unsigned int m_numb_output;
    unsigned int m_nepoch = 0;
    std::string m_loss;
    std::string m_out_file_name;
    std::vector<unsigned int> m_topology;
    std::vector<std::string> m_active_funcs;
    std::vector<double> m_input;
    std::vector<double> m_output;
    LayerDeque m_layer_deque;

    Network();
public:
    ~Network();

    class Exception : std::exception{
        public:
            std::string m_message;
            Exception(const char* message);
    };

    static Network* init_from_file(const std::string& in_file_name, const std::string& out_file_name);
    static Network* create(const unsigned int numb_input, const unsigned int numb_output, const std::vector<unsigned int>& topology, const std::string& out_file_name = "network.txt");
    std::vector<double> get_result(const std::vector<double>& input) const;
    void print(std::ostream& os) const;
    void save() const;
    void train_on_data(const std::vector<double>& input, const std::vector<double>& output);
};

using NetworkPtr = std::unique_ptr<Network>;