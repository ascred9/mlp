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


#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer.h"
#include "eigen-3.4.0/Eigen/Core"

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;


template <class LayerType>
class LayerDeque
{
private:
    std::vector<std::shared_ptr<LayerType>> m_layers;
    std::string m_loss_type;
    std::function<const Vector(const Vector&, const Vector&)> m_floss; // first is true val, second is estimation
    std::function<const Vector(const Vector&, const Vector&)> m_fploss; // first is true val, second is estimation
    unsigned int m_outsize;
    double m_step;
    double m_regulization_rate;

    std::vector<std::pair<Matrix, Vector>> get_gradient(const std::vector<double>& input, const std::vector<double>& output,
                                                        const std::vector<double>& weights) const;
    double get_L2_regulization() const;
    double get_L2_regulization_prime() const;
public:
    LayerDeque();
    ~LayerDeque();
    
    void add_layers(std::vector<unsigned int> topology);
    std::vector<double> calculate(const std::vector<double>& input) const;
    void clear();
    void generate_weights(const std::string& init_type);
    double get_step() const;
    void print(std::ostream& os) const;
    void set_active_funcs(const std::vector<std::string>& active_funcs);
    void set_layers(const std::vector<std::vector<double>>& matrices, const std::vector<std::vector<double>>& biases); // first is vector of matrices with weights, second is bias vector
    void set_loss_func(const std::string& loss_type);
    void set_regulization_rate(double regulization_rate);
    void set_step(const double step);
    double test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                const std::vector<std::vector<double>>& weights) const;
    void train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
               const std::vector<std::vector<double>>& weights, unsigned int batch_size = 1);
};
