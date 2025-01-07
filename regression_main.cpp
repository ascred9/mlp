#include <any>
#include <iostream>
#include <map>
#include <vector>
#include <random>
#include <iomanip>
#include <omp.h>

#include "include/network.h"
#include "include/bnetwork.h"
#include "include/gnetwork.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"

#include "RootDrawer.cpp"


int process(TString filename)
{
    //ROOT::EnableImplicitMT();

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
    //BayesianNetworkPtr net_ptr = std::make_unique<BayesianNetwork>();
    //GradientNetworkPtr net_ptr = std::make_unique<GradientNetwork>();
    //net_ptr->create(5, 1, {5, 10}, "build/bnetwork.txt"); return 1;
    //net_ptr->init_from_file("build/bnetwork_theta4.txt", "build/bnetwork_theta5.txt");
    //net_ptr->init_from_file("build/bnetwork.txt", "build/btest_theta.txt");
    //net_ptr->init_from_file("build/btest_theta.txt", "build/btest_theta.txt");
    //net_ptr->init_from_file("build/100_200_perpendicular_v4.txt", "build/100_200_perpendicular_v4.txt");
    //net_ptr->init_from_file("build/btest_theta.txt", "build/btest_theta2.txt");
    //net_ptr->init_from_file("build/bseam.txt", "build/btest.txt");
    //net_ptr->init_from_file("build/btest.txt", "build/btest.txt");
    //NetworkPtr net_ptr = std::make_unique<Network>();
    //net_ptr->create(5, 1, {5, 10, 5}, "build/test.txt"); return 1;
    net_ptr->init_from_file("build/test1.txt", "build/test1.txt");
    //net_ptr->init_from_file("build/test.txt", "build/test.txt");
    //net_ptr->init_from_file("build/500_1000_perpendicular.txt", "build/test2.txt");
    //net_ptr->init_from_file("build/test8.txt", "build/test9.txt");

    if (net_ptr == nullptr)
    {
        std::cout << "nullptr" << std::endl;
        return -1;
    }

    clock_t start, end;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    std::normal_distribution<> gaus(0., 100.0);

    int Nepoch = 94*3;//*15;
    int Nentries = tph->GetEntries();
    int batch_size = 1e3;
    int minibatch_size = 1;
    double T = .5;
    std::vector<std::vector<double>> in, out, weights;

    //TFile* wfile = new TFile("weights.root");
    //if (wfile->IsZombie())
    //  std::cout << "file read error" << std::endl;

    int count = 0;

    NovosibirskGenerator m_generator = NovosibirskGenerator(0., 0.4, 0.15);
    TH1F *hhh = new TH1F("hhh", "hhh", 300, -3, 3);

    std::map<int, double> max; // rho - key, max weight - value
    for (int i=0; i < Nentries * 0.8; i++)
    {
        tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;
        //if (abs(simen-en)>200) continue;

        double n_th = abs(th - M_PI/2);
        //in.push_back({lxe, csi, n_th, phi, rho});
        in.push_back({lxe, csi, rho, n_th, phi});
        //out.push_back({((lxe+csi)/simen)});
        out.push_back({simen});

        double weight = 1;//std::exp((rho-51)/T);//std::exp(-abs(en-simen) / T); //std::exp((n_th-0.55)/T);//std::exp(- w/T) / max[int(rho+0.5)]; 
        weights.push_back({weight});

        //hhh->Fill((simen-150)/50+m_generator.generate());
        //double alpha = 0.814751 + 0.0502268 * 1. / (1. + std::exp((rho - 42.83) / 1.622));
        //double weight = std::exp(-abs(lxe + csi - alpha * simen) / T);
        //if (lxe + csi - alpha * simen < 0) // left tail has more events
        //    weight = pow(weight, 1./2);
    }

    //hhh->Draw();

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
    std::ofstream fout("result.txt");
    for (int i = Nentries * 0.8; i < Nentries; ++i)
    {
    	tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;
        //if (abs(simen-en)>200) continue;

        double n_th = abs(th - M_PI/2);

        //auto res = net_ptr->get_result({lxe, csi, n_th, phi, rho});
        auto res = net_ptr->get_result({lxe, csi, rho, n_th, phi});
        rec = res.at(0);
        rec_en = (lxe+csi)/rec;
        rec_lxe = lxe/(lxe+csi) * rec;
        rec_csi = csi/(lxe+csi) * rec;

        fout << simen << " " << rec << " " << lxe << " " << rec_lxe << " " << csi << " " << rec_csi << std::endl;
        t->Fill();
    }

    net_ptr->print(std::cout);
    DrawNet(net_ptr.get());

    tph->GetEntry(999);
    //DrawEvent(net_ptr.get(), {lxe, csi, abs(th - M_PI/2), phi, rho});
    DrawEvent(net_ptr.get(), {lxe, csi, rho, abs(th - M_PI/2), phi});
    tph->GetEntry(1313);
    DrawEvent(net_ptr.get(), {lxe, csi, rho, abs(th - M_PI/2), phi});
    tph->GetEntry(2425);
    DrawEvent(net_ptr.get(), {lxe, csi, rho, abs(th - M_PI/2), phi});
    tph->GetEntry(666);
    DrawEvent(net_ptr.get(), {lxe, csi, rho, abs(th - M_PI/2), phi});

    t->Write();
    tnet->Write();
    fout.close();
    end = clock();
    std::cout << "Processing Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;
    outfile->Close();

    return 0;
}

int main(int argc, char** argv)
{
    process("tph_100_200.root");
    return 0;
}
