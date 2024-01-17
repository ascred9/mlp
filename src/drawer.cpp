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

std::function<void(const NodePrimitive& p)> NodePrimitive::m_draw_func =
    [](const Primitive& p){std::cout << "Draw method isn't implemented! " << p.m_val << std::endl;};

NodePrimitive::NodePrimitive(float x, float y, float r, float v):
    m_x(x),
    m_y(y),
    m_r(r)
    {m_val = v;}

std::function<void(const ConnectionPrimitive& p)> ConnectionPrimitive::m_draw_func =
    [](const Primitive& p){std::cout << "Draw method isn't implemented! " << p.m_val << std::endl;};

ConnectionPrimitive::ConnectionPrimitive(NodePrimitive from, NodePrimitive to, float val):
    m_node_from(from),
    m_node_to(to)
    {m_val = val;}

PrimitiveDrawer::PrimitiveDrawer(const Network* net)
{
    // Divide x, y sizes from 0 to 1
    float layer_h = 1./net->get_topology().size();
    float r = 1.;
    for (const auto& v: net->get_topology())
        r = r > 1./(2*v) ? 1./(2.5*v) : r;

    // Fill vector of primitives
    for (int i = 0; i < net->get_topology().size(); i++)
    {
        for (int j = 0; j < net->get_topology().at(i); j++)
        {
            float layer_w = 1./net->get_topology().at(i);
            auto node_ptr = std::make_shared<NodePrimitive>(layer_w*(j+0.5), layer_h*(i+0.5), r, 0.0);
            m_primitives.emplace_back(node_ptr);
        }
    }

    auto it = m_primitives.begin();
    for (int il = 0; il < net->get_topology().size()-1; il++)
    {
        auto jt = std::next(it, net->get_topology().at(il));
        for (int j = 0; j < net->get_topology().at(il); j++, it++)
        {
            auto kt = jt;
            for (int k = 0; k < net->get_topology().at(il+1); k++, kt++)
            {
                auto conn_ptr = std::make_shared<ConnectionPrimitive>(*(NodePrimitive*)it->get(), *(NodePrimitive*)(kt->get()), 1);
                m_primitives.emplace_back(conn_ptr);
            }
        }
    }
}

void PrimitiveDrawer::draw() const
{
    for (const auto& primitive: m_primitives)
        primitive->draw();
}
