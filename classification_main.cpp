#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "include/network.h"

#include "TFile.h"
#include "TTree.h"

int process(TString filename)
{
    TFile* infile = new TFile(filename);
    if (infile->IsZombie())
    {
      std::cout << "file read error" << std::endl;
      return - 1;
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

    //NetworkPtr net_ptr( Network::create(6, 2, {10, 5}, "build/class_network.txt") ); return 1;
    //NetworkPtr net_ptr( Network::init_from_file("network.txt", "test.txt") );
    NetworkPtr net_ptr( Network::init_from_file("build/class_network.txt", "build/class_test.txt") );
    //NetworkPtr net_ptr( Network::init_from_file("build/test.txt", "build/test.txt") );
    if (net_ptr == nullptr)
        return -1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0., 1.0);
    std::normal_distribution<> gaus(0., 10.0);

    clock_t start, end;
    int Nepoch = 1*94;
    int Nentries = tph->GetEntries();
    int batch_size = 1; //100;
    std::vector<std::vector<double>> in, out, weights;

    TFile* wfile = new TFile("weights.root");
    if (wfile->IsZombie())
    {
      std::cout << "file read error" << std::endl;
      return - 1;
    }

    for (int i=0; i < Nentries * 0.8; i++)
    {
        tph->GetEntry(i);
        if (bgo != 0 || phi > 7 || th > 4 || abs(th-M_PI/2)>0.5) continue;

        double n_th = abs(th - M_PI/2);
        double rndm_en = dis(gen) * 100. + 700.;
        in.push_back({rndm_en, lxe, csi, n_th, phi, rho});
        double up = rndm_en - simen > 0? 1.0 : 0.0; 
        double down = rndm_en - simen > 0? 0.0 : 1.0; 
        out.push_back({up, down});

        double weight = 1;
        weights.push_back({weight, weight});
    }

    for (int iep = 0; iep < Nepoch; ++iep)
    {
    	start = clock();
        net_ptr->train(in, out, weights, batch_size);
        end = clock();
        std::cout << "Training Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;
    }
    net_ptr->save();

    start = clock();
    TFile* outfile = new TFile("out.root", "recreate");
    TTree* t = new TTree("tph", "tph");
    float rndm_en, up, down;
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
    t->Branch("rndm_en",&rndm_en);
    t->Branch("up",     &up);
    t->Branch("down",   &down);
    //std::ofstream fout("result.txt");
    for (int i = Nentries * 0.8; i < Nentries; ++i)
    {
    	tph->GetEntry(i);
        if (bgo != 0 || phi > 7 || th > 4 || abs(th-M_PI/2)>0.5) continue;
        double n_th = abs(th - M_PI/2);

        double bot = 700., top = 800.;
        rndm_en = 750.;
        for (int i=0; i<10; i++)
        {
            auto res = net_ptr->get_result({rndm_en, lxe, csi, n_th, phi, rho});
            up = res.at(0);
            down = res.at(1);
            if (up < 0.5)
            {
                bot = rndm_en;
                rndm_en = (top + bot) / 2.;
            }
            else
            {
                top = rndm_en;
                rndm_en = (top + bot) / 2.;
            }
            std::cout << rndm_en << " ";
        }
        std::cout << std::endl;

        t->Fill();
    }
    t->Write();
    //fout.close();
    end = clock();
    std::cout << "Processing Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;

    return 0;
}
