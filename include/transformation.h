/**
 * @file transformation.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-09-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

class Transformation
{
protected:
    std::function<double(double)> m_transf;
    std::function<double(double)> m_reverse_transf;
    double m_bt_limit;
    double m_up_limit;
    bool m_limits_set;

public:
    Transformation(); // set bot and up limits automatically, not recommended
    Transformation(double bt_limit, double up_limit);
    bool check_limits(const double data); // return true if limits are changed!
    virtual void set_config() = 0; // calculate the transformation variables
    double transform(const double var) const;
    double reverse_transform(const double var) const;
    virtual void print(std::ostream& os) const = 0; // print type, nuber of vars and vars value
};
using TransformationPtr = std::shared_ptr<Transformation>;

class LinearTransformation: public Transformation
{
/*
    Transform data to a distibutaion from -1 to 1 by the next way
    x in [a, b] -> x - (a+b)/2 in [-(b-a)/2, (b-a)/2] -> (x - (a+b)/2) / ((b-a)/2) in [-1, 1]
*/
private:
    double m_slope; // (b-a)/2
    double m_shift; // (a+b)/2

public:
    LinearTransformation() : Transformation(){};
    LinearTransformation(double bt_limit, double up_limit);
    virtual void set_config() override;
    virtual void print(std::ostream& os) const override;
};

class NormalTransformation: public Transformation
{
/*
    Transform data to a distibutaion from -(m-a)/s to +(b+m)/s by the next way
    x in [a, b] -> x - m in [a-m, b-m] -> (x - m) / (s) in [-(m-a)/s, (b+m)/s],
    where m is mean and s is standard dev
*/
private:
    double m_mean; // sum( x ) / n
    double m_dev;  // sum( (x-m_mean)^2 )/ (n-1)

public:
    NormalTransformation() : Transformation(){};
    NormalTransformation(double bt_limit, double up_limit);
    void set_mean(double mean) {m_mean = mean;};
    void set_dev(double dev) {m_dev = dev;};
    virtual void set_config() override;
    virtual void print(std::ostream& os) const override;
};