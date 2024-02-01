/**
 * @file gradient_layer.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "../include/glayer.h"

GradientLayer::~GradientLayer()
{
}

void GradientLayer::add_gradient(const std::pair<Matrix, Vector>& dL, unsigned int batch_size)
{
    // Add factor 2 as by minibatch regulization is divided to 2 previously
    m_gradW += 2 * dL.first * 1./ batch_size;
    m_gradB += 2 * dL.second * 1./ batch_size;
    if (m_regulization_rate > 0.)
    {
        m_gradW += pow(m_regulization_rate, -2.) * get_matrixW() * 1./ batch_size * pow(2., -m_n_iteration);
        m_gradB += pow(m_regulization_rate, -2.) * get_vectorB() * 1./ batch_size * pow(2., -m_n_iteration);
    }

    if (m_minibatch_count == 0)
    {
        m_prevGradW = m_gradW;
        m_prevGradB = m_gradB;
        m_prevMatrixW = m_tempMatrixW;
        m_prevVectorB = m_tempVectorB;
        m_minibatch_count++;
        update_weights(m_diff_step);
    }
    else if (m_minibatch_count == 1)
        m_minibatch_count++;
        
}

void GradientLayer::update_weights(double step)
{
    if (m_minibatch_count != 1 && m_minibatch_count != 2)
        throw std::invalid_argument("minibatch value isn't 2: " + std::to_string(m_minibatch_count));
    
    if (m_minibatch_count == 2)
    {
        // Scale gradient by second deriv for gradients
        Matrix sdevMatrixW;
        sdevMatrixW.array() = (m_gradW - m_prevGradW).array() / (m_tempMatrixW - m_prevMatrixW).array();
        bool isnan = !sdevMatrixW.allFinite();
        if (isnan)
            sdevMatrixW.setZero();
        m_gradW.array() = 0.5 * (m_gradW + m_prevGradW).array() *
            (Matrix::Constant(m_size, m_out_size, 1.).array() +
                pow(m_diff_step/m_dregulization_rate, 2) * sdevMatrixW.array());
        
        Vector sdevVectorB;
        sdevVectorB.array() = (m_gradB - m_prevGradB).array() / (m_tempVectorB- m_prevVectorB).array();
        isnan = !sdevVectorB.allFinite();
        if (isnan)
            sdevVectorB.setZero();
        m_gradB.array() = 0.5 * (m_gradB + m_prevGradB).array() *
            (Vector::Constant(1, m_out_size, 1.).array() +
                pow(m_diff_step/m_dregulization_rate, 2) * sdevVectorB.array());

        //std::cout << "W: " << m_gradW << std::endl << "B: " << m_gradB << std::endl;
        //std::cout << "W: " << sdevMatrixW << std::endl << "B: " << sdevVectorB << std::endl;
        //int k;
        //std::cin >> k;

        // Set previous matrices
        m_matrixW = m_prevMatrixW;
        m_vectorB = m_prevVectorB;
        // Reset control value
        m_minibatch_count = 0;
        m_n_iteration++;
    }

    double ep = m_adagrad_rate > 0. ? 1e-8 : 1.;

    // Update speed
    double speed_boost = 1.;
    if (m_n_iteration < 1./(1-m_viscosity_rate))
        speed_boost = 1./(1-m_viscosity_rate);

    m_speedW += speed_boost * (1-m_viscosity_rate) * m_gradW;
    m_speedB += speed_boost * (1-m_viscosity_rate) * m_gradB;

    // Update memory
    double memory_boost = 1.;
    if (m_n_iteration < 1./(1-m_adagrad_rate))
        memory_boost = 1./(1-m_adagrad_rate);

    if (m_adagrad_rate > 0.)
    {
        m_memoryW.array() += memory_boost * (1-m_adagrad_rate) * m_gradW.array().pow(2);
        m_memoryB.array() += memory_boost * (1-m_adagrad_rate) * m_gradB.array().pow(2);
    }

    // Update weight matrix
    m_matrixW.array() -= step * m_speedW.array() /
      (m_memoryW.array() +
        Matrix::Constant(m_size, m_out_size, ep).array()).sqrt();
    m_vectorB.array() -= step * m_speedB.array() /
      (m_memoryB.array() +
        Vector::Constant(1, m_out_size, ep).array()).sqrt();

    // Decrease speed for the next iteration
    m_speedW *= m_viscosity_rate;
    m_speedB *= m_viscosity_rate;

    // Decrease memory for the next iteration
    if (m_adagrad_rate > 0)
    {
        m_memoryW *= m_adagrad_rate;
        m_memoryB *= m_adagrad_rate;
    }

    // Reset accumulated gradient
    m_gradW *= 0;
    m_gradB *= 0;
}