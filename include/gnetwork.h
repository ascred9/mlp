/**
 * @file gnetwork.h
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

class GradientNetwork: public Network
{
protected:
    virtual void add_layers() override;

public:
    GradientNetwork(): Network(){};
    ~GradientNetwork();
};

using GradientNetworkPtr = std::unique_ptr<GradientNetwork>;
