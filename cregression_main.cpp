#include <any>
#include <iostream>
#include <map>
#include <vector>
#include <random>
#include <iomanip>

#include "include/network.h"
#include "include/bnetwork.h"
#include "include/gnetwork.h"
#include "include/cnetwork.h"

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

    ConnectedNetworkPtr net_ptr1 = std::make_unique<ConnectedNetwork>();
    //net_ptr1->create(5, 1, {5, 5}, "build/1cnetwork.txt"); return 1;
    ConnectedNetworkPtr net_ptr2 = std::make_unique<ConnectedNetwork>();
    ConnectedNetworkPtr net_ptr3 = std::make_unique<ConnectedNetwork>();
    ConnectedNetworkPtr net_ptr4 = std::make_unique<ConnectedNetwork>();
    NetworkPtr net_ptr5 = std::make_unique<Network>();
    //net_ptr2->create(4, 2, {5, 5}, "build/2cnetwork.txt"); return 1;
    net_ptr1->init_from_file("build/1cnetwork.txt", "build/1connected.t");
    net_ptr2->init_from_file("build/2cnetwork.txt", "build/2connected.t");
    net_ptr3->init_from_file("build/1cnetwork.txt", "build/3connected.t");
    net_ptr4->init_from_file("build/2cnetwork.txt", "build/4connected.t");
    net_ptr5->init_from_file("build/1cnetwork.txt", "build/5connected.t");
    net_ptr1->set_forward_net(net_ptr2.get());
    net_ptr2->set_forward_net(net_ptr3.get());
    net_ptr3->set_forward_net(net_ptr4.get());

    if (net_ptr1 == nullptr || net_ptr2 == nullptr)
    {
        std::cout << "nullptr" << std::endl;
        return -1;
    }

    clock_t start, end;

    int Nepoch = 32; //2*94;
    int Nentries = tph->GetEntries();
    int batch_size = 1;
    int minibatch_size = 1;
    std::vector<std::vector<double>> in1, out1, in2, out2, in3, out3, in4, out4;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-2.0, 0.0);

    int count = 0;

    for (int i=0; i < Nentries * 0.8; i++)
    {
        tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;

        double n_th = abs(th - M_PI/2);
        double n_rho = rho + dis(gen); 
        in1.push_back({lxe, csi, n_th, phi, n_rho});
        out1.push_back({simen});
        in2.push_back({simen, n_th, phi, n_rho});
        out2.push_back({lxe, csi});
        in3.push_back({lxe, csi, n_th, phi, n_rho});
        out3.push_back({simen});
        in4.push_back({simen, n_th, phi, n_rho});
        out4.push_back({lxe, csi});
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
    net_ptr1->set_spectator_popfunc(popfunc);

    //net_ptr1->train(Nepoch, in1, out1, batch_size, minibatch_size);
    net_ptr1->train_chain(Nepoch, {in1, in2, in3, in4}, {out1, out2, out3, out4}, 1, 1, 0.5);
    net_ptr1->save();
    net_ptr2->save();
    net_ptr3->save();
    net_ptr4->save();

    start = clock();
    float rec, rec_en, rec_lxe, rec_csi, weight;
    float rr_lxe, rr_csi;
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
    t->Branch("rr_lxe", &rr_lxe);
    t->Branch("rr_csi", &rr_csi);
    t->Branch("weight", &weight);
    std::ofstream fout("result.txt");

    std::vector<std::vector<double>> in5, out5;

    for (int i = Nentries * 0.8; i < Nentries; ++i)
    {
    	tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;
        double n_th = abs(th - M_PI/2);
        double n_rho = rho + dis(gen); 

        auto res = net_ptr3->get_result({lxe, csi, n_th, phi, n_rho});
        rec = res.at(0);
        rec_en = (lxe+csi)/rec;
        rec_lxe = lxe/(lxe+csi) * rec;
        rec_csi = csi/(lxe+csi) * rec;

        auto res2 = net_ptr4->get_result({simen, n_th, phi, n_rho});
        rr_lxe = res2.at(0);
        rr_csi = res2.at(1);

        in5.push_back({rr_lxe, rr_csi, n_th, phi, n_rho});
        out5.push_back({simen});

        fout << simen << " " << rec << " " << lxe << " " << rec_lxe << " " << csi << " " << rec_csi << std::endl;
        //t->Fill();
    }

    net_ptr5->train(Nepoch, in5, out5, 1, 1);
    for (int i = Nentries * 0.8; i < Nentries; ++i)
    {
    	tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;
        double n_th = abs(th - M_PI/2);
        double n_rho = rho + dis(gen); 

        auto res2 = net_ptr4->get_result({simen, n_th, phi, n_rho});
        rr_lxe = res2.at(0);
        rr_csi = res2.at(1);
        auto res = net_ptr5->get_result({res2.at(0), res2.at(1), n_th, phi, n_rho});
        rec = res.at(0);
        rec_en = (lxe+csi)/rec;
        rec_lxe = lxe/(lxe+csi) * rec;
        rec_csi = csi/(lxe+csi) * rec;

        t->Fill();
    }

    net_ptr3->print(std::cout);
    DrawNet(net_ptr3.get());

    tph->GetEntry(999);
    DrawEvent(net_ptr3.get(), {lxe, csi, abs(th - M_PI/2), phi, rho});

    t->Write();
    tnet->Write();
    fout.close();
    end = clock();
    std::cout << "Processing Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;
    outfile->Close();

    return 0;
}
