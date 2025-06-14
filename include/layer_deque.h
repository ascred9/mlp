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
#include <fstream>
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
#include "glayer.h"
#include "kde.h"
#include "linearLS.h"
#include "kstest.h"


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
    double m_dropout_rate;
    double m_regulization_rate;
    double m_viscosity_rate;
    std::map<std::string, double*> m_pars_map; // collect all rate to the one container

    double m_alpha = 0;
    mutable Vector m_ema;

    Vector m_addition_gradient;

    std::vector<std::pair<Matrix, Vector>> get_gradient(const std::vector<double>& input, const std::vector<double>& output,
                                                        const std::vector<double>& weights) const;
    double get_regulization() const;

    void prepare_batch(const std::vector<std::vector<double>>& input,
                       const std::vector<std::vector<double>>& output,
                       unsigned int id, unsigned int batch_size);

    bool m_useZeroSlope = false;
    std::vector<std::array<double, 4>> m_ls_data;

    bool m_useKDE = false;
    std::unique_ptr<KDE> m_kde_local;
    std::unique_ptr<KDE> m_kde_global;

    bool m_useBinningKDE = true;
    static const int m_num = 20;
    std::vector<std::unique_ptr<KDE>> m_kdes;
    std::vector<int> m_ids;
    double m_kde_left[m_num];
    double m_kde_right[m_num];
    int m_nums_kde_left[m_num];
    int m_nums_kde_right[m_num];

    bool m_useKS = false;
    std::unique_ptr<KStest> m_ks;

    bool m_useBinningZeroMean = false;
    double m_means[m_num];
    double m_std_left[m_num];
    double m_std_right[m_num];
    int m_nums[m_num];
    int m_nums_left[m_num];
    int m_nums_right[m_num];

    bool m_neighbourSum = false;
    int m_kNeighbours = 10;
    double m_lastResByNeighbours = 0;
    double m_sigmaByNeighbours = 0;
    std::string m_savedNeighbours = "neighbours.dat";
    std::vector<int> m_sortedIndeces;
    std::vector<int> m_sortedPlaces;

    bool m_correlation = false;
    double m_corr;
    double m_s_mean;
    double m_r_sigma;
    double m_s_sigma;

    bool m_neigh = false;
    int m_kNeigh = 20;
    double m_sum;

    NovosibirskGenerator m_generator = NovosibirskGenerator(0., 1, 0.15);

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
    double get_dropout_rate() const {return m_dropout_rate;};
    void print(std::ostream& os) const;
    bool read_layer(std::istream& fin, int layer_id);
    void set_active_funcs(const std::vector<std::string>& active_funcs);
    void set_layers(const std::vector<std::vector<double>>& matrices, const std::vector<std::vector<double>>& biases); // first is vector of matrices with weights, second is bias vector
    void set_loss_func(const std::string& loss_type);
    void set_adagrad_rate(double adagrad_rate);
    void set_dropout_rate(double dropout_rate);
    void set_regulization_rate(double regulization_rate);
    void set_viscosity_rate(double viscosity_rate);
    void set_step(const double step);
    std::array<double, 3> test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
                const std::vector<std::vector<double>>& weights) const;
    void train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output,
               const std::vector<std::vector<double>>& weights, unsigned int batch_size = 1, unsigned int minibatch_size = 1);
    Vector train_event(const std::vector<double>& input, const std::vector<double>& output); // return Vector of dL/dx for this event, x is input
    void set_addition_gradient(const Vector& addition_gradient) {m_addition_gradient = addition_gradient;};
    
    const std::vector<std::vector<double>> get_calculatedX(const std::vector<double>& input) const;
    const std::vector<std::vector<double>> get_calculatedZ(const std::vector<double>& input) const;
};

