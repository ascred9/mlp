/**
 * @glayer.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "eigen-3.4.0/Eigen/Core"

#include "layer.h"

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;


class GradientLayer: public Layer
{
    friend class LayerDeque;

protected:
    Matrix m_prevMatrixW;
    Vector m_prevVectorB;
    Matrix m_prevGradW;
    Vector m_prevGradB;
    int m_minibatch_count = 0;
    double m_diff_step = 1e-5;
    double m_dregulization_rate = 3e-5;

public:
    GradientLayer(unsigned int size): Layer(size){};
    ~GradientLayer();
    virtual void add_gradient(const std::pair<Matrix, Vector>& dL, unsigned int batch_size) override;
    virtual void update_weights(double step) override;
};
