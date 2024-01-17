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
#include <memory>
#include <string>
#include <vector>

#include "network.h"

struct Primitive
{
private:
    static std::function<void(const Primitive& p)> m_draw_func;
public:
    float val = 0;
    virtual void print() const {std::cout << "Primitive" << std::endl;};
    static void set_draw_func(const std::function<void(const Primitive& p)>& draw_func) {m_draw_func = draw_func;};
    void draw() const {if (m_draw_func != nullptr) m_draw_func(*this);};
};

struct NodeDrawPrimitive: public Primitive
{
public:
    float x, y;
    float radii;
    virtual void print() const override {std::cout << "Node" << std::endl;};
};

struct ConnectionDrawPrimitive: public Primitive
{
public:
    NodeDrawPrimitive node_from;
    NodeDrawPrimitive node_to;
    virtual void print() const override {std::cout << "Connection" << std::endl;};
};

class PrimitiveDrawer
{
private:
    std::vector<Primitive> m_primitives;
public:
    PrimitiveDrawer(const Network* net); //calculate node positions
    const std::vector<Primitive> get_primitives() const {return m_primitives;};
    void draw() const;
    void draw_event(const std::vector<float>& input); // calculate all node
};