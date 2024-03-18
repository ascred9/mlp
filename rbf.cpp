#include <any>
#include <iostream>
#include <map>
#include <vector>
#include <random>
#include <iomanip>

#include "TFile.h"
#include "TTree.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"

int process_centers()
{
    TFile* mfile = new TFile("centers.root");
    if (mfile->IsZombie())
    {
        std::cout << "file read error" << std::endl;
        return -1;
    }
    
    TMatrixD* centers = (TMatrixD*)mfile->Get("centers");
    
    TFile* infile = new TFile("tph_data.root");
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

    clock_t start, end;

    int Nentries = tph->GetEntries();
    std::cout << "Entries: " << Nentries << std::endl;
    double T = 10.;
    std::vector<std::vector<double>> in;
    std::vector<double> out;

    std::vector<double> input;
    auto glxe = [](double lxe){return (lxe-434)/148.8;};
    auto gcsi = [](double csi){return (csi-254.5)/152.5;};
    auto lth = [](double th){return (th-1.571/2)/1.571;};
    auto lphi = [](double phi){return (phi-6.285/2)/6.285;};
    auto grho = [](double rho){return (rho-39.96)/2.85;};
    auto lsim = [](double simen){return (simen-750.)/50.;};

    int Ncenters = centers->GetNrows();
    start = clock();
    std::cout << "Start filling" << std::endl;
    for (int i = 0; i < 0.8*Nentries; i++)
    {
        tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;

        double n_th = abs(th - M_PI/2);
        input = {glxe(lxe), gcsi(csi), grho(rho)};//lth(n_th)};//, lphi(phi), grho(rho)};
        in.push_back(input);
        out.push_back(lsim(simen));
    }
    std::cout << "Size: " << in.size() << std::endl;
    int N = in.size();

    auto f = [T](const std::vector<double>& in1, const std::vector<double>& in2){
        double sum = 0;
        for (int i = 0; i < in1.size(); i++)
            sum += pow(in1.at(i) - in2.at(i), 2);
        return std::exp(-sum/T);
    };

    std::cout << "Start matrix filling" << std::endl;
    TMatrixD PhiT(Ncenters, N);
    TMatrixD Y(N, 1);
    for (int i = 0; i < N; i++)
    {
        double sum = 0;
        for (int j = 0; j < Ncenters; j++)
        {
            std::vector<double> center = {(*centers)(j, 0), (*centers)(j, 1), (*centers)(j, 2)};
            PhiT(j, i) = f(in.at(i), center);
            sum += f(in.at(i), center);
        }

        Y(i, 0) = out.at(i) * sum;
    }

    std::cout << "Start multiplying" << std::endl;
    TMatrixD PhiTPhi(Ncenters, Ncenters);
    PhiTPhi.MultT(PhiT, PhiT);
    std::cout << "Start inverting" << std::endl;
    TMatrixDSym PhiTPhiSym(Ncenters, PhiTPhi.GetMatrixArray());
    TMatrixDSym InvertPhiTPhiSym = PhiTPhiSym.Invert();
    //TMatrixD InvertPhiTPhi(Ncenters, Ncenters, InvertPhiTPhiSym.GetMatrixArray());
    //std::cout << InvertPhiTPhi.GetNrows() << "x" << InvertPhiTPhi.GetNcols() << std::endl;
    std::cout << "Calculate pseudo inverting" << std::endl;
    TMatrixD PseudoInvertPhi(Ncenters, N);
    PseudoInvertPhi.Mult(InvertPhiTPhiSym, PhiT);
    std::cout << "Calculate weights" << std::endl;
    TMatrixD W(Ncenters, 1);
    W.Mult(PseudoInvertPhi, Y);
    W.Print();

    TFile* outfile = new TFile("out.root", "recreate");
    TTree* t = new TTree("tph", "tph");

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
    
    std::cout << "Test" << std::endl;
    for (int i = 0.8*Nentries; i < Nentries; i++)
    {
        //std::cout << i << "/" << Nentries << std::endl;
    	tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;
        double n_th = abs(th - M_PI/2);
        input = {glxe(lxe), gcsi(csi), grho(rho)};//lth(n_th)};//, lphi(phi), grho(rho)};

        TMatrixD fx(1, Ncenters);
        double sum = 0;
        for (int j = 0; j < Ncenters; j++)
        {
            std::vector<double> center = {(*centers)(j, 0), (*centers)(j, 1), (*centers)(j, 2)};
            fx(0, j) = f(input, center);
            sum += f(input, center);
        }

        TMatrixD res(1, 1);
        res.Mult(fx, W);

        rec = res(0, 0) / sum * 50 + 750;
        rec_en = (lxe+csi)/rec;
        rec_lxe = lxe/(lxe+csi) * rec;
        rec_csi = csi/(lxe+csi) * rec;

        t->Fill();
    }

    t->Write();
    t->Draw("rec-simen:simen", "abs(rec-simen) < 100");
    end = clock();
    std::cout << "Processing Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;
    //outfile->Close();

    return 0;
}

int process()
{
    TFile* infile = new TFile("tph_data.root");
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

    clock_t start, end;

    int Nentries = tph->GetEntries();
    double T = .00001;
    std::vector<std::vector<double>> in;
    std::vector<double> out;

    std::vector<double> input;
    auto glxe = [](double lxe){return (lxe-434)/148.8;};
    auto gcsi = [](double csi){return (csi-254.5)/152.5;};
    auto lth = [](double th){return (th-1.571/2)/1.571;};
    auto lphi = [](double phi){return (phi-6.285/2)/6.285;};
    auto grho = [](double rho){return (rho-39.96)/2.85;};
    auto lsim = [](double simen){return (simen-750.)/50.;};

    int N = 2000;
    start = clock();
    for (int i=0; in.size() < N+1; i++)
    {
        tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;

        double n_th = abs(th - M_PI/2);
        input = {glxe(lxe), gcsi(csi), lth(n_th)};//, lphi(phi), grho(rho)};
        in.push_back(input);
        out.push_back(lsim(simen));
    }

    auto f = [T](const std::vector<double>& in1, const std::vector<double>& in2){
        double sum = 0;
        for (int i = 0; i < in1.size(); i++)
            sum += pow(in1.at(i) - in2.at(i), 2);
        return std::exp(-sum/T);
    };

    TMatrixDSym Phi(N);
    TMatrixD Y(N, 1);
    for (int i = 0; i < N; i++)
    {
        double sum = 0;
        for (int j = 0; j < N; j++)
        {
            Phi(i, j) = f(in.at(i), in.at(j));
            sum += f(in.at(i), in.at(j));
        }

        Y(i, 0) = out.at(i) * sum;
    }

    TMatrixDSym InvertPhi = Phi.Invert();
    TMatrixD W(N, 1);
    W.Mult(InvertPhi, Y);
    W.GetSub(0, 3, 0, 0).Print();

    TFile* outfile = new TFile("out.root", "recreate");
    TTree* t = new TTree("tph", "tph");

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

    for (int i = N; i < 10*N; i++)
    {
        //std::cout << i << "/" << Nentries << std::endl;
    	tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;
        double n_th = abs(th - M_PI/2);
        input = {glxe(lxe), gcsi(csi), lth(n_th)};//, lphi(phi), grho(rho)};

        TMatrixD fx(1, N);
        double sum = 0;
        for (int j = 0; j < N; j++)
        {
            fx(0, j) = f(input, in.at(j));
            sum += f(input, in.at(j));
        }

        TMatrixD res(1, 1);
        res.Mult(fx, W);

        rec = res(0, 0) / sum * 50 + 750;
        rec_en = (lxe+csi)/rec;
        rec_lxe = lxe/(lxe+csi) * rec;
        rec_csi = csi/(lxe+csi) * rec;

        t->Fill();
    }

    t->Write();
    t->Draw("rec-simen:simen", "abs(rec-simen) < 100");
    end = clock();
    std::cout << "Processing Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;
    //outfile->Close();

    return 0;
}
