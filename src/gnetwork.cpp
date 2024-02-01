/**
 * @file gnetwork.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "../include/gnetwork.h"

GradientNetwork::~GradientNetwork()
{
}

void GradientNetwork::add_layers()
{
    m_layer_deque.add_layers<GradientLayer>(m_topology);
}
