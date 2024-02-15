#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "TGraph2D.h"
#include "TMultiGraph.h"
#include "TRandom3.h"

#include "include/kmeans.h"

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

    return 1;
}