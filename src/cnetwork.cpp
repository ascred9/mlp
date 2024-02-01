/**
 * @file cnetwork.cpp
 * @author Aleksandr Semenov (ascred9@gmail.com), research scientist in HEP from BINP
 * @brief 
 * @version 0.1
 * @date 2024-02-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "../include/cnetwork.h"


void ConnectedNetwork::set_forward_net(ConnectedNetwork* forward_net)
{
    this->m_forward_net = forward_net;
    forward_net->m_backward_net = this;
    this->update_chain();
}

void ConnectedNetwork::update_chain()
{
    ConnectedNetwork* net = this;
    while (net->m_backward_net) // go to first netwrok
        net = net->m_backward_net;

    int id = 0;
    do // run through the chain and set id
    {
        net->m_network_id = id;
        net = net->m_forward_net;
        id++;
    }
    while (net);

    m_chain_size = id + 1;
}

double ConnectedNetwork::test_chain(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::vector<double>>>& output)
{
    update_chain();

    const unsigned int input_chain_size = input.size();
    const unsigned int output_chain_size = output.size();
    if (input_chain_size == 0 || output_chain_size == 0)
        throw Exception("incorrect size of input/output chain size! it is zero!");

    if (input_chain_size != output_chain_size)
        throw Exception("size of input and output chain sizes are not equal");

    if (m_chain_size - m_network_id != input_chain_size)
        throw Exception("you start from wrong network");

    double sum = 0;
    ConnectedNetwork* net = this;
    for (int inet = 0; inet < input.size(); inet++)
    {
        sum += net->test(input.at(inet), output.at(inet));
        net = net->m_forward_net;
    }

    return sum;
}

void ConnectedNetwork::train_chain(int nepoch, const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::vector<double>>>& output,
                                   unsigned int batch_size, int minibatch_size, double split_mode)
{
    update_chain();
    const unsigned int input_chain_size = input.size();
    const unsigned int output_chain_size = output.size();
    if (input_chain_size == 0 || output_chain_size == 0)
        throw Exception("incorrect size of input/output chain size! it is zero!");

    if (input_chain_size != output_chain_size)
        throw Exception("size of input and output chain sizes are not equal");

    if (m_chain_size - m_network_id != input_chain_size)
        throw Exception("you start from wrong network");

    std::vector<std::vector<std::vector<double>>> norm_input;
    std::vector<std::vector<std::vector<double>>> norm_output;

    ConnectedNetwork* net = this;
    for (int inet = 0; inet < input.size(); inet++)
    {
        // Set config of input and output normalization transformation
        std::for_each(input.at(inet).begin(), input.at(inet).end(), [this, net](const std::vector<double>& vars){
            for(unsigned int idx = 0; idx < net->m_in_transf.size(); ++idx)
                net->m_in_transf.at(idx)->check_limits(vars.at(idx));
        });

        std::for_each(output.at(inet).begin(), output.at(inet).end(), [this, net](const std::vector<double>& vars){
            for(unsigned int idx = 0; idx < net->m_out_transf.size(); ++idx)
                net->m_out_transf.at(idx)->check_limits(vars.at(idx));
        });

        std::for_each(net->m_in_transf.begin(), net->m_in_transf.end(), [](TransformationPtr& pTransf){pTransf->set_config();});
        std::for_each(net->m_out_transf.begin(), net->m_out_transf.end(), [](TransformationPtr& pTransf){pTransf->set_config();});

        // Normalize input vectors
        std::vector<std::vector<double>> transf_input(input.at(inet).begin(), input.at(inet).end());
        std::for_each(transf_input.begin(), transf_input.end(),
            [this, net](std::vector<double>& in_vars){net->transform_input(in_vars);
        });
    
        // Normalize output vectors
        std::vector<std::vector<double>> transf_output(output.at(inet).begin(), output.at(inet).end());
        std::for_each(transf_output.begin(), transf_output.end(),
            [this, net](std::vector<double>& out_vars){net->transform_output(out_vars);
        });
        
        // Add it to vector
        norm_input.push_back(transf_input);
        norm_output.push_back(transf_input);

        net = net->m_forward_net;
    }

    int numb_events = norm_input.at(0).size();
    std::vector<std::vector<std::vector<double>>> train_input(norm_input.begin(), norm_input.begin() + int(split_mode * numb_events));
    std::vector<std::vector<std::vector<double>>> train_output(norm_output.begin(), norm_output.begin() + int(split_mode * numb_events));
    
    std::vector<std::vector<std::vector<double>>> test_input(norm_input.begin() + int(split_mode * numb_events), norm_input.end());
    std::vector<std::vector<std::vector<double>>> test_output(norm_output.begin() + int(split_mode * numb_events), norm_output.end());

    train_chain_input(nepoch, train_input, train_output, test_input, test_output, batch_size, minibatch_size);
}

void ConnectedNetwork::train_chain_input(const int nepoch, const std::vector<std::vector<std::vector<double>>>& train_input, const std::vector<std::vector<std::vector<double>>>& train_output,
                                         const std::vector<std::vector<std::vector<double>>>& test_input, const std::vector<std::vector<std::vector<double>>>& test_output,
                                         unsigned int batch_size, unsigned int minibatch_size)
{
    ConnectedNetwork* end = this;
    for (int id = m_network_id; id < m_chain_size-1; id++)
        end = end->m_forward_net;

    int numb_train_events = train_input.at(0).size();
    for (int iepoch = 0; iepoch < nepoch; iepoch++)
    {
        for (int ievent = 0; ievent < numb_train_events; ievent++)
        {
            ConnectedNetwork* net = end;
            for (int idata = train_input.size() - 1; idata > -1; idata--)
            {
                net->m_layer_deque.train_event(train_input.at(idata).at(ievent), train_output.at(idata).at(ievent));
                net = net->m_backward_net;
            }
        }
    }
}