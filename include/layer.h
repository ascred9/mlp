#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <functional>


#include "eigen-3.4.0/Eigen/Core"

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

class Layer
{
private:
    unsigned int m_size;
    unsigned int m_in_size;
    unsigned int m_out_size;

    std::function<double(double)> m_f;
    std::function<double(double)> m_fp;

    Matrix m_matrixW;
    double m_bias;

public:
    Layer(unsigned int size);
    ~Layer();

    const Vector calculate(const Vector& input) const;
    void generate_weights(const std::string& init_type);
    void print(std::ostream& os) const;
    void set_in_size(unsigned int in_size);
    void set_out_size(unsigned int out_size);
    void set_func(std::string func_name);
    const unsigned int& size() const;
};

using LayerPtr = std::shared_ptr<Layer>;

class LayerDeque
{
private:
    std::vector<LayerPtr> m_layers;

public:
    LayerDeque();
    ~LayerDeque();
    
    void add_layers(std::vector<unsigned int> topology);
    std::vector<double> calculate(const std::vector<double>& input) const;
    void clear();
    void generate_weights(const std::string& init_type);
    void print(std::ostream& os) const;
};
