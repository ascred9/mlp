/**
 * @file spectator.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2023-02-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <sstream>
#include <string>

class Spectator
{
protected:
    std::stringstream m_netstring;
    int m_nepoch;
    double m_loss_mean;
    double m_loss_stddev;
    std::map<std::string, double> m_net_pars;

public:
    virtual void pop(); // Specific implemetation for your containers
    void upload();
};
