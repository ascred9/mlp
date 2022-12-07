/**
 * @file bnetwork.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-12-04
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include "network.h"

class BayesianNetwork: public Network
{
protected:
    virtual void add_layers() override;

public:
    BayesianNetwork(): Network(){};
    ~BayesianNetwork();
};

using BayesianNetworkPtr = std::unique_ptr<BayesianNetwork>;
