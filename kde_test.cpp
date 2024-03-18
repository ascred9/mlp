#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TCanvas.h"

#include "include/network.h"
#include "include/kde.h"

int test_kde()
{
    TFile* infile = new TFile("tph_data.root");
    if (infile->IsZombie())
    {
        std::cout << "file read error" << std::endl;
        return -1;
    }

    NetworkPtr net_ptr = std::make_unique<Network>();
    net_ptr->init_from_file("build/btest_theta3.txt", "build/btest_theta4.txt");
    
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

    std::vector<std::vector<double>> input;
    std::vector<double> sim;
    int N = tph->GetEntries();
    //for (int i = 0; i < 1000; ++i)
    for (int i = 0; input.size() < 1000; i++)
    {
        tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;

        double n_th = abs(th - M_PI/2);
        input.push_back({lxe, csi, rho});
        sim.push_back((simen-750)/50.);
    }
        
    std::vector<std::vector<double>> reco;
    for (const auto& in: input)
    {
        auto res = net_ptr->get_result(in);
        res.at(0) = (res.at(0) - 750)/50.;
        reco.push_back(res);
    }

    KDE kde;
    kde.recalculate(reco);

    TH1F* hist_sim = new TH1F("hsim", "hsim", 100, -2, 2);
    TH1F* hist_exp = new TH1F("hexp", "hexp", 100, -2, 2);
    
    TMultiGraph* mg = new TMultiGraph("mg", "mg");
    TGraph* graph_sim = new TGraph();
    graph_sim->SetName("gsim");
    TGraph* graph_dsim = new TGraph();
    graph_dsim->SetName("gdsim");
    TGraph* graph_exp = new TGraph();
    graph_exp->SetName("gexp");
    TGraph* gr_dep = new TGraph();
    gr_dep->SetName("gdep");
    TGraph* gr_kde = new TGraph();
    gr_kde->SetName("gkde");

    double KL = 0;
    for (int i = 0; i < 1000; i++)
    {
        double rec = reco.at(i).front();
        graph_sim->AddPoint(rec, kde.m_expected_f(rec));
        graph_dsim->AddPoint(rec, kde.m_expected_df(rec));
        graph_exp->AddPoint(rec, kde.m_f.at(i));
        gr_dep->AddPoint(sim.at(i), rec - sim.at(i));
        hist_sim->Fill(sim.at(i));
        hist_exp->Fill(rec);
        gr_kde->AddPoint(rec, kde.get_val(i));
    
        KL += log(kde.m_f.at(i)/kde.m_expected_f(rec));
    }
    KL /= reco.size();
    std::cout << "KL: " << KL << std::endl;

    TCanvas* c = new TCanvas("c", "c", 900, 600);
    c->Divide(2, 2);
    c->cd(1);
    graph_sim->SetMarkerColor(kBlue);
    graph_dsim->SetMarkerColor(kBlack);
    graph_exp->SetMarkerColor(kRed);
    mg->Add(graph_sim, "AP");
    mg->Add(graph_dsim, "AP");
    mg->Add(graph_exp, "AP");
    mg->Draw("AP");

    c->cd(2);
    gr_kde->SetMarkerColor(kBlack);
    gr_kde->Draw("AP");

    c->cd(3);
    hist_sim->SetLineColor(kBlue);
    hist_sim->Draw();
    hist_exp->SetLineColor(kRed);
    hist_exp->Draw("same");

    c->cd(4);
    gr_dep->Draw("AP");

    return 0;
}