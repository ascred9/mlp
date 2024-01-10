/**
 * @file layer.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-09-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <ctime>
#include <random>

#include "eigen-3.4.0/Eigen/Core"

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;


class Layer
{
    friend class LayerDeque;

protected:
    unsigned int m_size;
    unsigned int m_in_size;
    unsigned int m_out_size;
    bool m_trainMode = false;

    // Learning status
    int m_n_iteration = 0;

    // Activization function
    std::function<double(double)> m_f;
    std::function<double(double)> m_fp;

    // Muscle of layer
    Matrix m_matrixW;
    Vector m_vectorB;
    Matrix m_tempMatrixW;
    Vector m_tempVectorB;
    Matrix m_gradW;
    Vector m_gradB;
    Matrix m_speedW;
    Vector m_speedB;
    Matrix m_memoryW;
    Vector m_memoryB;
    const double m_bias;

    // Generator for dropout
    std::mt19937 m_gen;
    std::uniform_real_distribution<> m_uniform{0., 1.};

    // Learning parameters
    double m_regulization_rate;
    double m_viscosity_rate;
    double m_adagrad_rate;
    double m_dropout_rate;

    const Vector calculate(const Vector& input) const;
    const Vector calculateZ(const Vector& inputX) const;
    const Vector calculateX(const Vector& inputZ) const;
    const Vector calculateXp(const Vector& inputZ) const;
    const Matrix& get_matrixW() const;
    const Vector& get_vectorB() const;
    const unsigned int& size() const; 
    void set_in_size(unsigned int in_size);
    void set_out_size(unsigned int out_size);
    void set_func(std::string func_name);
    void set_matrixW(const std::vector<double>& weights);
    void set_matrixW(const Matrix& weights);
    void set_vectorB(const std::vector<double>& biases);
    void set_vectorB(const Vector& biases);

public:
    Layer(unsigned int size);
    ~Layer();
    virtual void add_gradient(const std::pair<Matrix, Vector>& dL);
    virtual void generate_weights(const std::string& init_type);
    virtual double get_regulization() const;
    virtual void print(std::ostream& os) const;
    virtual bool read(std::istream& fin);
    virtual void reset_layer(const std::map<std::string, double*>& learning_pars);
    virtual void update();
    virtual void update_weights(double step);
};

using LayerPtr = std::shared_ptr<Layer>;
