/**
 * @byesian_layer.h
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2022-11-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <ctime>
#include <random>

#include "eigen-3.4.0/Eigen/Core"

#include <iostream>
#include "layer.h"

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;


class BayesianLayer: public Layer
{
    friend class LayerDeque;

protected:
    Matrix m_devMatrixW;
    Vector m_devVectorB;
    Matrix m_epsilonW;
    Vector m_epsilonB;
    Matrix m_tempMatrixW;
    Vector m_tempVectorB;

    Matrix m_gradDW;
    Vector m_gradDB;

    std::mt19937 m_gen;
    std::normal_distribution<> m_gaus{0., 1.};

    virtual const Matrix& get_matrixW() const override;
    virtual const Vector& get_vectorB() const override;

    void set_devMatrixW(const std::vector<double>& devWeights);
    void set_devMatrixW(const Matrix& devWeights);
    void set_devVectorB(const std::vector<double>& devBiases);
    void set_devVectorB(const Vector& devBiases);

public:
    BayesianLayer(unsigned int size): Layer(size){ m_gen.seed(std::time(nullptr));};
    ~BayesianLayer();
    virtual void add_gradient(double reg, const std::pair<Matrix, Vector>& dL) override;
    virtual void generate_weights(const std::string& init_type) override;
    virtual void print(std::ostream& os) const override;
    virtual bool read(std::istream& fin) override;
    virtual void reset_grads() override;
    virtual void update() override;
    virtual void update_weights(double step) override;
};
