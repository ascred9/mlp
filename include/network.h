#include <iostream>
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
    std::vector<unsigned int> m_topology;
    std::vector<double> m_input;
    std::vector<double> m_output;
    std::string m_out_file_name;
    LayerDeque m_layers;

    Network();
public:
    ~Network();

    class Exception : std::exception{
        public:
            std::string m_message;
            Exception(const char* message);
    };

    void init_from_file(const std::string& in_file_name, const std::string& out_file_name);
    static Network* create(const unsigned int numb_input, const unsigned int numb_output, const std::vector<unsigned int>& topology, const std::string& out_file_name = "network.txt");
    std::vector<double> get_result(const std::vector<double>& input) const;
    void print(std::ostream& os) const;
};

using NetworkPtr = std::unique_ptr<Network>;