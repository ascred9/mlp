/**
 * @file network.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-09-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "layer.h"

# define M_PI           3.14159265358979323846  /* pi */

class Network
{
private:
    unsigned int m_numb_input;
    unsigned int m_numb_output;
    unsigned int m_nepoch;
    std::string m_loss;
    std::string m_out_file_name;
    std::vector<unsigned int> m_topology;
    std::vector<std::string> m_active_funcs;
    std::vector<double> m_input;
    std::vector<double> m_output;
    LayerDeque m_layer_deque;
    std::vector<TransformationPtr> m_in_transf;
    std::vector<TransformationPtr> m_out_transf;

    Network();
    void add_in_transform(const std::string& type);
    void add_in_transform(const std::string& type, const std::vector<double>& vars);
    void add_out_transform(const std::string& type);
    void add_out_transform(const std::string& type, const std::vector<double>& vars);
    void transform_input(std::vector<double>& in_value) const; // transform input from initital range to range [-1, +1]
    void transform_output(std::vector<double>& out_value) const; // transform output from initial range to range [-1, +1]
    void reverse_transform_output(std::vector<double>& out_value) const; // reverse transform output from [-1, +1] to initial range
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
    double test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                const std::vector<std::vector<double>>& weights) const;
    double test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output) const;
    void train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
               const std::vector<std::vector<double>>& weights, unsigned int batch_size = 1, double split_mode = 0.5); // split mode is in [0, 1]
    void train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output, unsigned int batch_size = 1, double split_mode = 0.5); // split mode is in [0, 1]
};

using NetworkPtr = std::unique_ptr<Network>;
