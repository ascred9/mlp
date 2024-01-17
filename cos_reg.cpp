#include <any>
#include <iostream>
#include <map>
#include <vector>
#include <random>
#include <iomanip>

#include "include/network.h"
#include "include/bnetwork.h"

#include "TFile.h"
#include "TTree.h"
#include "Drawer.cpp"

int process()
{
    BayesianNetworkPtr net_ptr = std::make_unique<BayesianNetwork>();
    //net_ptr->create(1, 1, {10, 10}, "build/ncos.txt"); return 1;
    //net_ptr->init_from_file("build/bcos.txt", "build/btcos.txt");
    //net_ptr->init_from_file("build/btest.txt", "build/btest.txt");
    //NetworkPtr net_ptr = std::make_unique<Network>();
    //net_ptr->create(5, 1, {4}, "build/network.txt"); return 1;
    net_ptr->init_from_file("build/ncos.txt", "build/test.txt");

    if (net_ptr == nullptr)
    {
        std::cout << "nullptr" << std::endl;
        return -1;
    }

    clock_t start, end;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-3.0, 3.0);
    std::normal_distribution<> gaus(0., 100.0);

    Draw();

    int Nepoch = 2;//32;//219;//2e3;
    int Nentries = 100000;
    int batch_size = 10;
    int minibatch_size = 5;
    std::vector<std::vector<double>> in, out;

    int count = 0;

    for (int i=0; i < Nentries * 0.8; i++)
    {
        double x = dis(gen);
        in.push_back({x});
        out.push_back({cos(x)});
    }

    TFile* outfile = new TFile("out.root", "recreate");
    TTree* t = new TTree("tph", "tph");

    // Think about to shrink it
    TTree* tnet = new TTree("tnet", "tnet");
    std::string structura;
    int nepoch;
    float mean_loss, dev_loss, step, reg, visc, ada, drop;
    tnet->Branch("struct", &structura);
    tnet->Branch("nepoch", &nepoch);
    tnet->Branch("mean_loss", &mean_loss);
    tnet->Branch("dev_loss", &dev_loss);
    tnet->Branch("step", &step);
    tnet->Branch("reg", &reg);
    tnet->Branch("visc", &visc);
    tnet->Branch("ada", &ada);
    tnet->Branch("drop", &drop);
    std::function<void(const std::map<std::string, std::any>&)> popfunc = 
      [&](const std::map<std::string, std::any>& notebook){
        structura = std::any_cast<std::string>(notebook.at("struct")); 
        nepoch = std::any_cast<unsigned int>(notebook.at("nepoch")); 
        mean_loss = std::any_cast<double>(notebook.at("mean_loss")); 
        dev_loss = std::any_cast<double>(notebook.at("dev_loss")); 
        step = std::any_cast<double>(notebook.at("step")); 
        reg = std::any_cast<double>(notebook.at("regulization_rate")); 
        visc = std::any_cast<double>(notebook.at("viscosity_rate")); 
        ada = std::any_cast<double>(notebook.at("adagrad_rate"));
        drop = std::any_cast<double>(notebook.at("dropout_rate"));
        tnet->Fill();
    };
    net_ptr->set_spectator_popfunc(popfunc);


    net_ptr->train(Nepoch, in, out, batch_size, minibatch_size);
    net_ptr->save();

    start = clock();
    float x, cs, rec;
    t->Branch("x",&x);
    t->Branch("cs",&cs);
    t->Branch("rec",&rec);
    std::ofstream fout("result.txt");
    for (int i = Nentries * 0.8; i < Nentries; ++i)
    {
        x = dis(gen);
        cs = cos(x);
        auto res = net_ptr->get_result({x});
        rec = res.at(0);
        t->Fill();
    }
    t->Write();
    tnet->Write();
    fout.close();
    end = clock();
    std::cout << "Processing Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;
    net_ptr->print(std::cout);

    return 0;
}
