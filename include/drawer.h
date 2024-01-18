/**
 * @file drawer.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-01-17
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <iostream>
#include <list>
#include <memory>
#include <string>

#include "network.h"

class Primitive
{
public:
    float m_val;
    virtual void print() const {std::cout << "Primitive" << std::endl;};
    virtual void draw() const = 0;
};


class NodePrimitive: public Primitive
{
private:
    static std::function<void(const NodePrimitive& p)> m_draw_func;
public:
    float m_x, m_y, m_r;
    NodePrimitive(float x = 0, float y = 0, float r = 0, float v = 0);
    virtual void print() const override {std::cout << "Node " << m_x << " " << m_y << std::endl;};
    static void set_draw_func(const std::function<void(const NodePrimitive& p)>& draw_func) {m_draw_func = draw_func;};
    virtual void draw() const override {if (m_draw_func != nullptr) m_draw_func(*this);};
};

class ConnectionPrimitive: public Primitive
{
private:
    static std::function<void(const ConnectionPrimitive& p)> m_draw_func;
public:
    float m_x1, m_y1, m_r1;
    float m_x2, m_y2, m_r2;
    ConnectionPrimitive(NodePrimitive* from, NodePrimitive* to, float v = 0);
    virtual void print() const override {std::cout << "Connection " << m_x1 << " " << m_x2 << std::endl;};
    static void set_draw_func(const std::function<void(const ConnectionPrimitive& p)>& draw_func) {m_draw_func = draw_func;};
    virtual void draw() const override {if (m_draw_func != nullptr) m_draw_func(*this);};
};

class PrimitiveDrawer
{
private:
    std::list<std::shared_ptr<Primitive>> m_primitives;
    std::list<std::shared_ptr<Primitive>> m_order;
    void read_weights(const Network* net);
    float m_min_weight, m_max_weight;
    std::vector<float> m_weights;
    std::vector<float> m_biases;
public:
    PrimitiveDrawer(const Network* net); //calculate node positions
    const std::list<std::shared_ptr<Primitive>> get_primitives() const {return m_primitives;};
    void draw() const;
    void draw_event(const std::vector<float>& input); // calculate all node
};