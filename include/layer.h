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
    friend class LayerDeque;
private:
    unsigned int m_size;
    unsigned int m_in_size = 0.;
    unsigned int m_out_size = 0.;

    std::function<double(double)> m_f;
    std::function<double(double)> m_fp;

    Matrix m_matrixW;
    Vector m_vectorB;
    const double m_bias = 1.;

    const Vector calculate(const Vector& input) const;
    const Vector calculateZ(const Vector& inputX) const;
    const Vector calculateX(const Vector& inputZ) const;
    const Vector calculateXp(const Vector& inputZ) const;
    void generate_weights(const std::string& init_type);
    const Matrix& get_matrixW() const;
    const Vector& get_vectorB() const;
    void print(std::ostream& os) const;
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

class LayerDeque
{
private:
    std::vector<LayerPtr> m_layers;
    std::string m_loss_type;
    std::function<const Vector(const Vector&, const Vector&)> m_floss; // first is true val, second is estimation
    std::function<const Vector(const Vector&, const Vector&)> m_fploss; // first is true val, second is estimation
    double m_step = 0.5;

    std::vector<std::pair<Matrix, Vector>> get_gradient(const std::vector<double>& input, const std::vector<double>& output) const;
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
    void set_step(const double step);
    double test(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output, unsigned int batch_size = 1) const;
    void train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output, unsigned int batch_size = 1);
};
