#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "TGraph2D.h"
#include "TMultiGraph.h"
#include "TRandom3.h"
#include "TVectorD.h"
#include "TCanvas.h"

#include "include/kmeans.h"
#include "include/pca.h"

int test()
{
    TRandom3 rand(999);

    int N = 100000;
    std::vector<std::vector<double>> input;
    for (int i = 0; i < N; ++i)
    {
        if (rand.Uniform() < 0.5)
            input.push_back({rand.Gaus(-1, 1), rand.Gaus(-1, 1), rand.Gaus(-1, 1)});
        else
            input.push_back({rand.Gaus(1, 1), rand.Gaus(1, 1), rand.Gaus(-1, 1)});
    }

    KMeans kmeans(20);
    kmeans.calculate(input);

    TGraph2D* graph = new TGraph2D();
    for (const auto& item: input)
        graph->AddPoint(item.at(0), item.at(1), item.at(2));

    TGraph2D* markers = new TGraph2D();
    auto centers = kmeans.get_centers();
    for (const auto& center: centers)
        markers->AddPoint(center.at(0), center.at(1), center.at(2));
    
    graph->SetMarkerSize(2);
    markers->SetMarkerColor(kRed);
    markers->SetMarkerStyle(34);

    graph->Draw("AP");
    markers->Draw("same p0");

    return 1;
}

int process_real()
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

    auto glxe = [](double lxe){return (lxe-434)/148.8;};
    auto gcsi = [](double csi){return (csi-254.5)/152.5;};
    auto lth = [](double th){return (th-1.571/2)/1.571;};
    auto lphi = [](double phi){return (phi-6.285/2)/6.285;};
    auto grho = [](double rho){return (rho-39.96)/2.85;};
    auto lsim = [](double simen){return (simen-750.)/50.;};

    std::vector<std::vector<double>> input;
    int N = tph->GetEntries();
    for (int i = 0; i < N; ++i)
    {
        tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;

        double n_th = abs(th - M_PI/2);
        input.push_back({glxe(lxe), gcsi(csi), grho(rho)});
    }

    int Ncenters = 1000;
    KMeans kmeans(Ncenters);
    kmeans.calculate(input);

    TGraph2D* graph = new TGraph2D();
    for (const auto& item: input)
        graph->AddPoint(item.at(0), item.at(1), item.at(2));

    TGraph2D* markers = new TGraph2D();
    auto centers = kmeans.get_centers();
    for (const auto& center: centers)
        markers->AddPoint(center.at(0), center.at(1), center.at(2));
    
    graph->SetMarkerSize(2);
    graph->GetXaxis()->SetTitle("elxe");
    graph->GetYaxis()->SetTitle("ecsi");
    graph->GetZaxis()->SetTitle("rho");
    markers->SetMarkerColor(kRed);
    markers->SetMarkerStyle(34);
    markers->GetXaxis()->SetTitle("elxe");
    markers->GetYaxis()->SetTitle("ecsi");
    markers->GetZaxis()->SetTitle("rho");

    graph->Draw("AP");
    markers->Draw("same p0");

    TFile* fout = new TFile("centers.root", "RECREATE");
    TMatrixD* matrix = new TMatrixD(Ncenters, 3);
    for (int i = 0; i < Ncenters; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            (*matrix)(i, j) = centers.at(i).at(j);
        }
    }
    matrix->Print();

    matrix->Write("centers");
    fout->Close();

    return 1;
}

int test_pca()
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

    auto glxe = [](double lxe){return (lxe-434)/148.8;};
    auto gcsi = [](double csi){return (csi-254.5)/152.5;};
    auto lth = [](double th){return (th-1.571/2)/1.571;};
    auto lphi = [](double phi){return (phi-6.285/2)/6.285;};
    auto grho = [](double rho){return (rho-39.96)/2.85;};
    auto lsim = [](double simen){return (simen-750.)/50.;};

    std::vector<std::vector<double>> input;
    int N = tph->GetEntries();
    for (int i = 0; i < 0.8*N; ++i)
    {
        tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;

        double n_th = abs(th - M_PI/2);
        input.push_back({glxe(lxe), gcsi(csi), grho(rho)});
    }

    PCA pca;
    auto res = pca.calculate(input);

    std::cout << "values" << std::endl;
    for (const auto& p: res.first)
    {
        std::cout << p << std::endl;
    }

    std::cout << "vectors" << std::endl;
    for (const auto& p: res.second)
    {
        for (const auto& v: p)
            std::cout << v << " ";

        std::cout << std::endl;
    }

    std::cout << "scalar" << std::endl;
    auto v1 = res.second.at(0);
    auto v2 = res.second.at(2);
    std::cout << v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2] << std::endl;

    TGraph2D *graph2d_init = new TGraph2D();
    graph2d_init->SetTitle("initial");
    TGraph2D *graph2d_trns = new TGraph2D();
    graph2d_trns->SetTitle("after");
    for (int i = 0.2*N; i < N; ++i)
    {
        tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;

        double n_th = abs(th - M_PI/2);
        std::vector<double> in{glxe(lxe), gcsi(csi), grho(rho)};
        graph2d_init->AddPoint(in.at(0), in.at(1), in.at(2));
        pca.transform(in);
        graph2d_trns->AddPoint(in.at(0), in.at(1), in.at(2));
    }

    TCanvas* c = new TCanvas("c", "c", 900, 600);
    c->Divide(2);
    c->cd(1);
    graph2d_init->Draw();
    c->cd(2);
    graph2d_trns->Draw();

    return 0;
}