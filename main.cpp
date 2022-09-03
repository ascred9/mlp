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
    NetworkPtr net_ptr( Network::init_from_file("test.txt", "test.txt") );
    if (net_ptr == nullptr)
        return -1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    clock_t start, end;
    int Nepoch = 60;
    int Nevent = 5e4;
    for (int iep = 0; iep < Nepoch; ++iep)
    {
        start = clock();
        std::vector<std::vector<double>> in, out;
        for (int i=0; i < Nevent; i++)
        {
            double x = 2 * dis(gen);
            double y = dis(gen) + x;
            double z = x * x * cos(y);
            in.push_back({x, y});
            out.push_back({z});
        }
        net_ptr->train(in, out);
        end = clock();
        std::cout << "Training Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;
    }
    net_ptr->save();

    start = clock();
    std::ofstream fout("cos.txt");
    for (int i = 0; i < Nevent; ++i)
    {
        double x = 2 * dis(gen);
        double y = dis(gen) + x;
        double real = x * x * cos(y);
        double output = net_ptr->get_result({x, y}).at(0);
        fout << real << " " << output << " diff: " << real-output << std::endl;
    }
    fout.close();
    end = clock();
    std::cout << "Processing Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;

    return 0;
}