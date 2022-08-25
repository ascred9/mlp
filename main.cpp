#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "network.h"

int main()
{
    std::cout << "Hello world! Sancho" << std::endl;
    //NetworkPtr net_ptr( Network::create(2, 1, {9}, "network.txt") ); return 1;
    //NetworkPtr net_ptr( Network::init_from_file("network.txt", "test.txt") );
    //NetworkPtr net_ptr( Network::init_from_file("xor_sigmoid.txt", "test.txt") );
    //NetworkPtr net_ptr( Network::init_from_file("network.txt", "test.txt") );
    NetworkPtr net_ptr( Network::init_from_file("test.txt", "test.txt") );

    // TODO: test XOR
    //net_ptr->get_result({1., 1.});
    //net_ptr->get_result({1., 0.});
    //net_ptr->get_result({0., 1.});
    //net_ptr->get_result({0., 0.});

    //net_ptr->train_on_data({1., 0.}, {1.});
    //net_ptr->train_on_data({1., 1.}, {0.});
    //net_ptr->train_on_data({0., 1.}, {1.});
    //net_ptr->train_on_data({0., 0.}, {0.});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    clock_t start, end;
    start = clock();
    int Nepoch = 5e4;
    for (int i=0; i < Nepoch; i++)
    {
        double x = dis(gen);
        double y = dis(gen);
        double z = x * y;
        net_ptr->train_on_data({x, y}, {z});
    }
    net_ptr->save();
    end = clock();
    std::cout << "Training Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;

    start = clock();
    std::ofstream fout("cos.txt");
    for (int i = 0; i < Nepoch; ++i)
    {
        double x = dis(gen);
        double y = dis(gen);
        double real = x * y;
        double output = net_ptr->get_result({x, y}).at(0);
        fout << real << " " << output << " diff: " << real-output << std::endl;
    }
    fout.close();
    end = clock();
    std::cout << "Processing Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;

    return 0;
}