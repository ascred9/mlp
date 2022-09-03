/**
 * @file transformation.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-09-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "transformation.h"

Transformation::Transformation():
    m_bt_limit(0),
    m_up_limit(0),
    m_limits_set(false)
{
}

Transformation::Transformation(double bt_limit, double up_limit):
    m_bt_limit(bt_limit),
    m_up_limit(up_limit),
    m_limits_set(true)
{
}

bool Transformation::check_limits(const double data)
{
    bool limits_changed = false;
    if (m_limits_set)
    {
        return limits_changed;
    }
    // Check limits
    if (m_bt_limit > data)
    {
        m_bt_limit = data;
        limits_changed = true;
    }
    if (m_up_limit < data)
    {
        m_up_limit = data;
        limits_changed = true;
    }

    return limits_changed;
}

double Transformation::transform(const double var) const
{   
    if (!m_transf)
        throw std::invalid_argument("the normalization transformation is empty!"); // TODO: make an global Exception static class

    return m_transf(var);
}

double Transformation::reverse_transform(const double var) const
{   
    if (!m_reverse_transf)
        throw std::invalid_argument("the reverse normalization transformation is empty!"); // TODO: make an global Exception static class

    return m_reverse_transf(var);
}

LinearTransformation::LinearTransformation(double bt_limit, double up_limit) : Transformation(bt_limit, up_limit)
{
    m_shift = 0.5 * (m_up_limit - m_bt_limit) + m_bt_limit; // avoiding stack overflow
    m_slope = 0.5 * (m_up_limit - m_bt_limit);

    m_transf = [this](double x){return (x - m_shift)/m_slope;};
    m_reverse_transf = [this](double y){return y * m_slope + m_shift;};
}

void LinearTransformation::set_config()
{
    m_shift = 0.5 * (m_up_limit - m_bt_limit) + m_bt_limit; // avoiding stack overflow
    m_slope = 0.5 * (m_up_limit - m_bt_limit);

    m_transf = [this](double x){return (x - m_shift)/m_slope;};
    m_reverse_transf = [this](double y){return y * m_slope + m_shift;};
}
    
void LinearTransformation::print(std::ostream& os) const
{
    os << "linear ";
    if (m_limits_set)
    {
        os << "2 " << m_bt_limit << " " << m_up_limit;
    }
    else
    {
        os << "0";
    }
    os << std::endl;
}