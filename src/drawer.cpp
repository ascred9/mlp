/**
 * @file drawer.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-01-17
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "../include/drawer.h"

std::function<void(const Primitive& p)> Primitive::m_draw_func = [](const Primitive& p){std::cout << "Draw method isn't implemented! " << p.val << std::endl;};

PrimitiveDrawer::PrimitiveDrawer(const Network* net)
{
    std::cout << "Size: " << net->get_topology().size() << std::endl;
    NodeDrawPrimitive node;
    node.radii = 10;
    m_primitives.push_back(node);
    // Divide x, y sizes from 0 to 1
    // Fill vector of primitives
}

void PrimitiveDrawer::draw() const
{
    for (const auto& primitive: m_primitives)
        primitive.draw();
}
