#include <any>
#include <iostream>
#include <map>
#include <vector>
#include <random>
#include <iomanip>

#include "include/network.h"
#include "include/bnetwork.h"
#include "include/gnetwork.h"

#include "TFile.h"
#include "TTree.h"

#include "RootDrawer.cpp"

int process(TString filename)
{
    TFile* infile = new TFile(filename);
    if (infile->IsZombie())
    {
        std::cout << "file read error" << std::endl;
        return -1;
    }
    
    TTree* tph = (TTree*)infile->Get("tph");
    float simen;
    int fc;
    float en, en0, lxe, csi, bgo, th, phi, rho;
    tph->SetBranchAddress("simen",	&simen);
    tph->SetBranchAddress("fc",		&fc);
    tph->SetBranchAddress("en",		&en);
    tph->SetBranchAddress("en0",	&en0);
    tph->SetBranchAddress("lxe",	&lxe);
    tph->SetBranchAddress("csi",	&csi);
    tph->SetBranchAddress("bgo",	&bgo);
    tph->SetBranchAddress("th",		&th);
    tph->SetBranchAddress("phi",	&phi);
    tph->SetBranchAddress("rho",	&rho);

    NetworkPtr net_ptr = std::make_unique<Network>();
    net_ptr->init_from_file("build/net_experiment.txt", "build/net_experiment2.txt");
    //net_ptr->create(5, 1, {10, 10, 10}, "build/network.txt"); return 1;

    if (net_ptr == nullptr)
    {
        std::cout << "nullptr" << std::endl;
        return -1;
    }

    clock_t start, end;

    int Nepoch = 32;//18*32; //2*94;
    int Nentries = tph->GetEntries();
    int batch_size = 1000;
    int minibatch_size = 1;
    std::vector<std::vector<double>> in, out, weights;

    int count = 0;
    
    std::vector<int> indexes(Nentries);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::shuffle(indexes.begin(), indexes.end(), std::mt19937 {std::random_device{}()});

    for (int i=0; i < Nentries * 0.8; i++)
    {
        tph->GetEntry(indexes.at(i));
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;
        if (abs(simen-en)>100) continue;

        double n_th = abs(th - M_PI/2);
        in.push_back({lxe, csi, rho, n_th, phi});
        out.push_back({simen});

        double weight = 1;
        weights.push_back({weight});
    }

    TFile* outfile = new TFile("out_experiment.root", "recreate");
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


    net_ptr->train(Nepoch, in, out, weights, batch_size, minibatch_size);
    net_ptr->save();

    start = clock();
    float rec, rec_en, rec_lxe, rec_csi, weight;
    t->Branch("simen",  &simen);
    t->Branch("en",     &en);
    t->Branch("en0",    &en0);
    t->Branch("lxe",    &lxe);
    t->Branch("csi",    &csi);
    t->Branch("bgo",    &bgo);
    t->Branch("th",     &th);
    t->Branch("phi",    &phi);
    t->Branch("rho",    &rho);
    t->Branch("fc",     &fc);
    t->Branch("rec",    &rec);
    t->Branch("rec_en", &rec_en);
    t->Branch("rec_lxe",&rec_lxe);
    t->Branch("rec_csi",&rec_csi);
    t->Branch("weight", &weight);
    for (int i = Nentries * 0.8; i < Nentries; ++i)
    {
    	tph->GetEntry(indexes.at(i));
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;
        if (abs(simen-en)>200) continue;

        double n_th = abs(th - M_PI/2);

        auto res = net_ptr->get_result({lxe, csi, rho, n_th, phi});
        rec = res.at(0);
        rec_en = (lxe+csi)/rec;
        rec_lxe = lxe/(lxe+csi) * rec;
        rec_csi = csi/(lxe+csi) * rec;

        t->Fill();
    }

    net_ptr->print(std::cout);
    DrawNet(net_ptr.get());

    tph->GetEntry(999);
    DrawEvent(net_ptr.get(), {lxe, csi, rho, abs(th - M_PI/2), phi});

    t->Write();
    tnet->Write();
    end = clock();
    std::cout << "Processing Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;
    outfile->Close();

    return 0;
}
