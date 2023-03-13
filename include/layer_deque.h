/**
 * @file layer_deque.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-11-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "eigen-3.4.0/Eigen/Core"

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

#include "layer.h"
#include "blayer.h"


class LayerDeque
{
private:
    std::vector<std::shared_ptr<Layer>> m_layers;
    std::string m_loss_type;
    std::function<const Vector(const Vector&, const Vector&)> m_floss; // first is true val, second is estimation
    std::function<const Vector(const Vector&, const Vector&)> m_fploss; // first is true val, second is estimation
    unsigned int m_outsize;
    double m_step;
    double m_adagrad_rate;
    double m_regulization_rate;
    double m_viscosity_rate;
    std::map<std::string, double*> m_pars_map; // collect all rate to ine container

    double m_alpha = 0.9;
    mutable Vector m_ema;

    std::vector<std::pair<Matrix, Vector>> get_gradient(const std::vector<double>& input, const std::vector<double>& output,
                                                        const std::vector<double>& weights) const;
    std::vector<std::pair<Matrix, Vector>> get_gradient_reg(const std::vector<double>& weights) const;
    double get_regulization() const;
public:
    LayerDeque();

    ~LayerDeque();
    
    template <class LayerT>
    typename std::enable_if<std::is_base_of<Layer, LayerT>::value, void>::type
    add_layers(std::vector<unsigned int> topology);

    std::vector<double> calculate(const std::vector<double>& input) const;
    void clear();
    void generate_weights(const std::string& init_type);
    double get_step() const {return m_step;};
    double get_regulization_rate() const {return m_regulization_rate;};
    double get_viscosity_rate() const {return m_viscosity_rate;};
    double get_adagrad_rate() const {return m_adagrad_rate;};
    void print(std::ostream& os) const;
    bool read_layer(std::istream& fin, int layer_id);
    void set_active_funcs(const std::vector<std::string>& active_funcs);
    void set_layers(const std::vector<std::vector<double>>& matrices, const std::vector<std::vector<double>>& biases); // first is vector of matrices with weights, second is bias vector
    void set_loss_func(const std::string& loss_type);
    void set_adagrad_rate(double adagrad_rate);
    void set_regulization_rate(double regulization_rate);
    void set_viscosity_rate(double viscosity_rate);
    void set_step(const double step);
    std::pair<double, double> test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                const std::vector<std::vector<double>>& weights) const;
    void train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
               const std::vector<std::vector<double>>& weights, unsigned int batch_size = 1, unsigned int minibatch_size = 1);
};

