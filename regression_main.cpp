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

    //NetworkPtr net_ptr( Network::create(5, 1, {10, 5}, "build/network.txt") ); return 1;
    //NetworkPtr net_ptr( Network::init_from_file("network.txt", "test.txt") );
    NetworkPtr net_ptr( Network::init_from_file("build/network.txt", "build/test.txt") );
    //NetworkPtr net_ptr( Network::init_from_file("build/test.txt", "build/test.txt") );
    if (net_ptr == nullptr)
        return -1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    std::normal_distribution<> gaus(0., 10.0);

    clock_t start, end;
    int Nepoch = 1*94;
    int Nentries = tph->GetEntries();
    int batch_size = 4;
    double T = 3;
    std::vector<std::vector<double>> in, out, weights;

    TFile* wfile = new TFile("weights.root");
    if (wfile->IsZombie())
    {
      std::cout << "file read error" << std::endl;
      return - 1;
    }

    TTree* wt = (TTree*)wfile->Get("weights");
    int nev;
    float w;
    wt->SetBranchAddress("nev",    &nev);
    wt->SetBranchAddress("weight", &w);
    int count = 0;

    std::map<int, double> max; // rho - key, max weight - value
    for (int i=0; i < Nentries * 0.8; i++)
    {
        tph->GetEntry(i);
        if (bgo != 0 || phi > 7 || th > 4 || abs(th-M_PI/2)>0.5) continue;

        while(true)
        {
            if (count > wt->GetEntries())
                return - 1;

            wt->GetEntry(count);
            if ( i == nev)
                break;

            ++count;
        }

        double n_th = abs(th - M_PI/2);
        in.push_back({lxe, csi, n_th, phi, rho});
        out.push_back({(csi+lxe)/simen});

        double weight = std::exp(-w/T);
        if (max[int(rho+0.5)] < weight)
            max[int(rho+0.5)] = weight;

        //double alpha = 0.814751 + 0.0502268 * 1. / (1. + std::exp((rho - 42.83) / 1.622));
        //double weight = std::exp(-abs(lxe + csi - alpha * simen) / T);
        //if (lxe + csi - alpha * simen < 0) // left tail has more events
        //    weight = pow(weight, 1./2);
    }

    // Fill and normalize weights
    count = 0;
    for (int i=0; i < Nentries * 0.8; i++)
    {
        tph->GetEntry(i);
        if (bgo != 0 || phi > 7 || th > 4 || abs(th-M_PI/2)>0.5) continue;
      
        while(true)
        {
            if (count > wt->GetEntries())
                return - 1;

            wt->GetEntry(count);
            if ( i == nev)
                break;
            
            ++count;
        }
 
        double weight = std::exp(- w/T) / max[int(rho+0.5)]; 
        //weights.push_back({weight});
        weights.push_back({1.});
        //std::cout << i << " " << (lxe+csi)/simen << " " << weight << " " << rho << std::endl;
    }

    for (auto it=max.begin(); it != max.end(); ++it)
        std::cout << it->first << " " << it->second << std::endl;

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
    float rec, rec_lxe, rec_csi, weight;
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
    t->Branch("rec_lxe",&rec_lxe);
    t->Branch("rec_csi",&rec_csi);
    t->Branch("weight", &weight);
    std::ofstream fout("result.txt");
    for (int i = Nentries * 0.8; i < Nentries; ++i)
    {
    	tph->GetEntry(i);
        if (bgo != 0 || phi > 7 || th > 4 || abs(th-M_PI/2)>0.5) continue;
        double n_th = abs(th - M_PI/2);

        auto res = net_ptr->get_result({lxe, csi, n_th, phi, rho});
        rec = res.at(0);
        rec_lxe = lxe/(lxe+csi) * rec;
        rec_csi = csi/(lxe+csi) * rec;

        while(true)
        {
            if (count > wt->GetEntries())
                return - 1;

            wt->GetEntry(count);
            if ( i == nev)
                break;

            ++count;
        }

        weight = std::exp(-w/T)/max.at(int(rho+0.5)); 

        fout << simen << " " << rec << " " << lxe << " " << rec_lxe << " " << csi << " " << rec_csi << std::endl;
        t->Fill();
    }
    t->Write();
    fout.close();
    end = clock();
    std::cout << "Processing Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;

    return 0;
}