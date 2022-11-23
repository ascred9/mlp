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


#include <memory>

#include "eigen-3.4.0/Eigen/Core"

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;


class Layer
{
    template <class LayerType>
    friend class LayerDeque;

    unsigned int m_size;
    unsigned int m_in_size;
    unsigned int m_out_size;

    std::function<double(double)> m_f;
    std::function<double(double)> m_fp;

    Matrix m_matrixW;
    Vector m_vectorB;
    const double m_bias;

    const Vector calculate(const Vector& input) const;
    const Vector calculateZ(const Vector& inputX) const;
    const Vector calculateX(const Vector& inputZ) const;
    const Vector calculateXp(const Vector& inputZ) const;
    void generate_weights(const std::string& init_type);
    const Matrix& get_matrixW() const;
    const Vector& get_vectorB() const;
    virtual void print(std::ostream& os) const;
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
};

using LayerPtr = std::shared_ptr<Layer>;
