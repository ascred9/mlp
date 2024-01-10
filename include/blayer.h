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


#include "eigen-3.4.0/Eigen/Core"

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

    Matrix m_gradDW;
    Vector m_gradDB;
    Matrix m_speedDW;
    Vector m_speedDB;
    Matrix m_memoryDW;
    Vector m_memoryDB;

    std::normal_distribution<> m_gaus{0., 1.};

    std::function<const Matrix(const Matrix&)> m_Wsigma = [](const Matrix& rho){
        return log(rho.array().exp() + 1.);
    };
    std::function<const Matrix(const Matrix&)> m_pWsigma = [](const Matrix& rho){
        return rho.array().exp() / (rho.array().exp() + 1.);
    };

    std::function<const Vector(const Vector&)> m_Bsigma = [](const Vector& rho){
        return log(rho.array().exp() + 1.);
    };
    std::function<const Vector(const Vector&)> m_pBsigma = [](const Vector& rho){
        return rho.array().exp() / (rho.array().exp() + 1.);
    };

    void set_devMatrixW(const std::vector<double>& devWeights);
    void set_devMatrixW(const Matrix& devWeights);
    void set_devVectorB(const std::vector<double>& devBiases);
    void set_devVectorB(const Vector& devBiases);

public:
    BayesianLayer(unsigned int size): Layer(size){};
    ~BayesianLayer();
    virtual void add_gradient(const std::pair<Matrix, Vector>& dL) override;
    virtual void generate_weights(const std::string& init_type) override;
    virtual double get_regulization() const override;
    virtual void print(std::ostream& os) const override;
    virtual bool read(std::istream& fin) override;
    virtual void reset_layer(const std::map<std::string, double*>& learning_pars) override;
    virtual void update() override;
    virtual void update_weights(double step) override;
};
