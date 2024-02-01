/**
 * @file cnetwork.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "network.h"

// Attempt to implement Autoencoder
// In this moment batch_size and minibatch_size have to be nun
// Please, save the order of parameter
class ConnectedNetwork: public Network
{
protected:
    static int m_chain_size;
    int m_network_id = 0;
    ConnectedNetwork* m_forward_net = nullptr;
    ConnectedNetwork* m_backward_net = nullptr;

    void train_chain_input(const int nepoch, const std::vector<std::vector<std::vector<double>>>& train_input, const std::vector<std::vector<std::vector<double>>>& train_output,
                           const std::vector<std::vector<std::vector<double>>>& test_input, const std::vector<std::vector<std::vector<double>>>& test_output,
                           unsigned int batch_size, unsigned int minibatch_size);
    void update_chain();

public:
    ConnectedNetwork(): Network(){};
    ~ConnectedNetwork();

    void set_forward_net(ConnectedNetwork* forward_net);

    double test_chain(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::vector<double>>>& output);
    void train_chain(int nepoch, const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::vector<double>>>& output,
                     unsigned int batch_size, int minibatch_size, double split_mode); // split mode is in [0, 1]
};

using ConnectedNetworkPtr = std::unique_ptr<ConnectedNetwork>;
