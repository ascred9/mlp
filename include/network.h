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

#pragma once

#include <algorithm>
#include <any>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <vector>

#include "layer_deque.h"
#include "transformation.h"
#include "pca.h"

# define M_PI           3.14159265358979323846  /* pi */

class Network
{
protected:
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
    std::unique_ptr<PCA> m_pca;
    bool m_usePCA = false;

    virtual void add_layers();
    void add_in_transform(const std::string& type);
    void add_in_transform(const std::string& type, const std::vector<double>& vars);
    void add_out_transform(const std::string& type);
    void add_out_transform(const std::string& type, const std::vector<double>& vars);
    void transform_input(std::vector<double>& in_value) const; // transform input from initital range to range [-1, +1]
    void transform_output(std::vector<double>& out_value) const; // transform output from initial range to range [-1, +1]
    void reverse_transform_output(std::vector<double>& out_value) const; // reverse transform output from [-1, +1] to initial range

    virtual void train_input(const int nepoch, std::vector<std::vector<double>> train_input, std::vector<std::vector<double>> train_output,
                             std::vector<std::vector<double>> train_weights,
                             const std::vector<std::vector<double>>& test_input, const std::vector<std::vector<double>>& test_output,
                             const std::vector<std::vector<double>>& test_weights,
                             unsigned int batch_size, unsigned int minibatch_size);

    // To save network in training steps
    std::function<void(const std::map<std::string, std::any>& notebook)> m_spec_popfunc;
    std::function<void()> m_spec_upgrade;
    void pop(const std::array<double, 3>& epsilon) const;

public:
    Network();
    ~Network();

    class Exception : std::exception{
        public:
            std::string m_message;
            Exception(const char* message);
    };

    bool init_from_file(const std::string& in_file_name, const std::string& out_file_name);
    bool create(const unsigned int numb_input, const unsigned int numb_output, const std::vector<unsigned int>& topology, const std::string& out_file_name = "network.txt");
    std::vector<double> get_result(const std::vector<double>& input) const;
    void print(std::ostream& os) const;
    void save() const;
    double test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                const std::vector<std::vector<double>>& weights) const;
    double test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output) const;
    void train(const int nepoch, const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
               const std::vector<std::vector<double>>& weights, unsigned int batch_size = 1, unsigned int minibatch_size = 1, double split_mode = 0.5); // split mode is in [0, 1]
    void train(const int nepoch, const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
               unsigned int batch_size = 1, unsigned int minibatch_size = 1, double split_mode = 0.5); // split mode is in [0, 1]

    void set_spectator_popfunc(const std::function<void(std::map<std::string, std::any>)>& popfunc){ m_spec_popfunc = popfunc;}; // Specify implemetation for your containers
    void set_spectator_upfunc(std::function<void()>); // Specify implementation for your containers

    const std::vector<unsigned int> get_topology() const {return m_topology;}
    const std::vector<std::vector<double>> get_calculatedX(const std::vector<double>& input) const;
    const std::vector<std::vector<double>> get_calculatedZ(const std::vector<double>& input) const;
};

using NetworkPtr = std::unique_ptr<Network>;
