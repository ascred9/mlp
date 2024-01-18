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
{
    m_val = v;
}

std::function<void(const ConnectionPrimitive& p)> ConnectionPrimitive::m_draw_func =
    [](const Primitive& p){std::cout << "Draw method isn't implemented! " << p.m_val << std::endl;};

ConnectionPrimitive::ConnectionPrimitive(NodePrimitive* from, NodePrimitive* to, float v)
{
    m_val = v;
    m_x1 = from->m_x;
    m_y1 = from->m_y;
    m_r1 = from->m_r;
    m_x2 = to->m_x;
    m_y2 = to->m_y;
    m_r2 = to->m_r;
}

PrimitiveDrawer::PrimitiveDrawer(const Network* net)
{
    std::cout << "~~~ Draw Neural Network! ~~~" << std::endl;
    // Set min and max weight value
    read_weights(net);
    // Divide x, y sizes from 0 to 1
    float layer_h = 1./net->get_topology().size();
    float r = 1.;
    for (const auto& v: net->get_topology())
        r = r > 1./(2*v) ? 1./(2.5*v) : r;

    // Fill vector of primitives
    float node_id = 0;
    for (unsigned int i = 0; i < net->get_topology().size(); i++)
    {
        unsigned int layer_size = net->get_topology().at(i);
        for (unsigned int j = 0; j < layer_size; j++)
        {
            float layer_w = 1./layer_size;
            auto node_ptr = std::make_shared<NodePrimitive>(layer_w*(j+0.5), layer_h*(i+0.5), r, node_id == 0 ? 0.0 : m_biases.at(node_id-1));
            node_id++;
            m_primitives.emplace_back(node_ptr);
        }
    }

    auto it = m_primitives.begin();
    int conn_id = 0;
    for (unsigned int il = 0; il < net->get_topology().size()-1; il++)
    {
        auto jt = std::next(it, net->get_topology().at(il));
        for (unsigned int j = 0; j < net->get_topology().at(il); j++, it++)
        {
            auto kt = jt;
            for (unsigned int k = 0; k < net->get_topology().at(il+1); k++, kt++)
            {
                float val = (m_weights.at(conn_id) - m_min_weight) / (m_max_weight - m_min_weight);
                conn_id++;
                auto conn_ptr = std::make_shared<ConnectionPrimitive>((NodePrimitive*)it->get(), (NodePrimitive*)(kt->get()), val);
                m_primitives.emplace_back(conn_ptr);
            }
        }
    }

    // Copy and sort only connections
    m_order = m_primitives;
    m_order.sort([](auto p1, auto p2){
        //TODO: Finish with Node. Currently, below method doesn't work
        if (!(NodePrimitive*)p1.get() || !(NodePrimitive*)p2.get())
        {
            std::cout << "PP" << std::endl;
            p1->print();
            std::cout << std::endl;
            p2->print();
            return true;
        }
        return abs(p1->m_val-0.5) < abs(p2->m_val-0.5);
    });
}

void PrimitiveDrawer::read_weights(const Network* net)
{
    std::stringstream ss;
    net->print(ss);
    std::string str;
    while (ss >> str)
    {
        if (str == "layer")
        {
            ss >> str; // read [ scope
            while (ss >> str && str != "]")
                m_weights.push_back(std::stof(str));

            ss >> str; // read { scope
            while (ss >> str && str != "}")
                m_biases.push_back(std::stof(str));
        }
    }

    auto max_w = std::max_element(m_weights.begin(), m_weights.end());
    auto max_b = std::max_element(m_weights.begin(), m_weights.end());
    m_max_weight = std::max(*max_w, *max_b);

    auto min_w = std::min_element(m_weights.begin(), m_weights.end());
    auto min_b = std::min_element(m_weights.begin(), m_weights.end());
    m_min_weight = std::min(*min_w, *min_b);

    std::cout << "Finish reading! Sizes: " << m_weights.size() << " and " << m_biases.size() << std::endl;
    std::cout << "Min: " << m_min_weight << ", Max: " << m_max_weight << std::endl;
}

void PrimitiveDrawer::draw() const
{
    for (const auto& primitive: m_order)
        primitive->draw();
}
