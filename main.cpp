#include <iostream>
#include <vector>

#include "network.h"

int main()
{
    std::cout << "Hello world! Sancho" << std::endl;
    NetworkPtr net_ptr( Network::create(2, 1, {3}) );

    // TODO: test XOR
    net_ptr->get_result({1., 1.})[0];
    net_ptr->get_result({1., 0.})[0];
    net_ptr->get_result({0., 1.})[0];
    net_ptr->get_result({0., 0.})[0];

    return 0;
}