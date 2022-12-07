/**
 * @file bnetwork.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-12-04
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../include/bnetwork.h"

BayesianNetwork::~BayesianNetwork()
{
}

void BayesianNetwork::add_layers()
{
    m_layer_deque.add_layers<BayesianLayer>(m_topology);
}
