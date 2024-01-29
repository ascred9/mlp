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
class ConnectedNetwork: public Network
{
protected:
    const ConnectedNetwork* m_forward_net = nullptr;
    const ConnectedNetwork* m_backward_net = nullptr;

public:
    ConnectedNetwork(): Network(){};
    ~ConnectedNetwork();

    void set_forward_net(ConnectedNetwork* forward_net);
};

using ConnectedNetworkPtr = std::unique_ptr<ConnectedNetwork>;
