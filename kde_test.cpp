#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TLegend.h"

#include "include/network.h"
#include "include/kde.h"

int test_kde()
{
    //TFile* infile = new TFile("tph_data.root");
    TFile* infile = new TFile("tph_100_200.root");
    if (infile->IsZombie())
    {
        std::cout << "file read error" << std::endl;
        return -1;
    }

    NetworkPtr net_ptr = std::make_unique<Network>();
    //net_ptr->init_from_file("build/btest_theta3.txt", "build/btest_theta4.txt");
    net_ptr->init_from_file("build/100_200_perpendicular_v9.txt", "build/100_200_perpendicular_v9.txt");
    
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
    int size = 10000;
    //for (int i = 0; i < 1000; ++i)
    for (int i = 0; input.size() < size; i++)
    {
        tph->GetEntry(i);
        if (phi > 7 || th > 4 || rho < 37 || abs(th-M_PI/2)>0.57 || bgo > 0) continue;
        if (abs(simen-en)>200) continue;

        double n_th = abs(th - M_PI/2);
        input.push_back({lxe, csi, rho, n_th, phi});
        sim.push_back((simen-150)/50.);
    }
        
    std::vector<double> reco;
    for (const auto& in: input)
    {
        auto res = net_ptr->get_result(in);
        res.at(0) = (res.at(0) - 150)/50.;
        reco.push_back(res.at(0));
    }

    clock_t start, end;
    start = clock();
    KDE kde;
    kde.set_verbose();
    kde.recalculate(reco);
    end = clock();
    std::cout << "Timedelta: " << std::setprecision(9) << double(end-start) / double(CLOCKS_PER_SEC) << std::setprecision(9) << " sec" << std::endl;

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

    TMultiGraph* mg2 = new TMultiGraph("mg2", "mg2");
    TGraph* gr_kde = new TGraph();
    gr_kde->SetName("gkde");
    TGraph* gr_log = new TGraph();
    gr_log->SetName("glog");
    TGraph* gr_plog = new TGraph();
    gr_plog->SetName("gplog");

    double KL = 0;
    for (int i = 0; i < size; i++)
    {
        double rec = reco.at(i);
        if (abs(rec) > 2)
            continue;
        graph_sim->AddPoint(rec, kde.m_expected_f(rec));
        graph_dsim->AddPoint(rec, kde.m_expected_df(rec));
        graph_exp->AddPoint(rec, kde.m_f.at(i));
        gr_dep->AddPoint(sim.at(i), rec - sim.at(i));
        hist_sim->Fill(sim.at(i));
        hist_exp->Fill(rec);
        gr_kde->AddPoint(rec, kde.get_gradient(i));
        gr_log->AddPoint(rec, log(kde.m_f.at(i)/kde.m_expected_f(rec)));
        gr_plog->AddPoint(rec, kde.m_f.at(i)*log(kde.m_f.at(i)/kde.m_expected_f(rec)));
    
        KL += log(kde.m_f.at(i)/kde.m_expected_f(rec));
    }
    KL /= reco.size();
    std::cout << "KL: " << KL << std::endl;

    TCanvas* c = new TCanvas("c", "c", 900, 600);
    c->Divide(2, 2);
    c->cd(1)->SetGrid();
    c->cd(1);
    graph_sim->SetMarkerColor(kBlue);
    graph_dsim->SetMarkerColor(kBlack);
    graph_exp->SetMarkerColor(kRed);
    graph_sim->SetLineColor(kBlue);
    graph_dsim->SetLineColor(kBlack);
    graph_exp->SetLineColor(kRed);
    mg->Add(graph_sim, "AP");
    mg->Add(graph_dsim, "AP");
    mg->Add(graph_exp, "AP");
    mg->Draw("AP");
    TLegend* legend1 = new TLegend(0.1, 0.7, 0.3, 0.9);
    legend1->AddEntry(graph_sim, "smoothed sim", "l");
    legend1->AddEntry(graph_dsim, "deriv sim", "l");
    legend1->AddEntry(graph_exp, "reco", "l");
    legend1->Draw();

    c->cd(2)->SetGrid();
    c->cd(2);
    gr_kde->SetMarkerColor(kBlack);
    gr_log->SetMarkerColor(kRed);
    gr_plog->SetMarkerColor(kBlue);
    gr_kde->SetLineColor(kBlack);
    gr_log->SetLineColor(kRed);
    gr_plog->SetLineColor(kBlue);
    mg2->Add(gr_kde, "AP");
    mg2->Add(gr_log, "AP");
    mg2->Add(gr_plog, "AP");
    mg2->Draw("AP");
    TLegend* legend2 = new TLegend(0.1, 0.7, 0.3, 0.9);
    legend2->AddEntry(gr_log, "ln(p(xi)/q(xi))", "l");
    legend2->AddEntry(gr_kde, "dKL/dxi", "l");
    legend2->AddEntry(gr_plog, "p * ln(p/q)", "l");
    legend2->Draw();

    c->cd(3);
    hist_sim->SetLineColor(kBlue);
    hist_exp->SetLineColor(kRed);
    hist_exp->Draw();
    hist_sim->Draw("same");

    c->cd(4)->SetGrid();
    c->cd(4);
    gr_dep->Draw("AP");

    TMultiGraph* mg3 = new TMultiGraph("mg3", "mg3");
    TGraph* graph_cdf_act = new TGraph();
    graph_cdf_act->SetName("cdf actual");
    TGraph* graph_cdf_exp = new TGraph();
    graph_cdf_exp->SetName("cdf expected");
    TGraph* graph_cdf_diff = new TGraph();
    graph_cdf_diff->SetName("cdf diff");
    std::multiset<double> sorted_reco(reco.begin(), reco.end());

    auto func = [](double x) {
        double a = sqrt(2) * 0.2;
        return -0.141047*a*pow(2.71828, -pow((1-x)/a, 2)) + 0.141047*a*pow(2.71828, -pow((1+x)/a, 2))
                + (0.25*x - 0.25) * std::erf((1-x)/a) + (0.25*x + 0.25) * std::erf((1+x)/a) + 0.5;
    };

    int count = 0;
    for (auto rec: sorted_reco)
    {
        count++;
        graph_cdf_act->AddPoint(rec, count * 1./sorted_reco.size());
        graph_cdf_exp->AddPoint(rec, func(rec));
        graph_cdf_diff->AddPoint(rec, abs(count * 1./sorted_reco.size() - func(rec)));
    }

    TCanvas* c2 = new TCanvas("c2", "c2", 900, 900);
    c2->Divide(2);
    c2->cd(1)->SetGrid();
    c2->cd(1);
    mg3->Add(graph_cdf_act, "AP");
    mg3->Add(graph_cdf_exp, "AP");
    graph_cdf_exp->SetMarkerColor(kRed);
    mg3->Draw("AP");

    c2->cd(2)->SetGrid();
    c2->cd(2);
    graph_cdf_diff->Draw("AP");
    std::cout << "Kolmogorov-Smirnov test: " << graph_cdf_diff->GetMaximum() << std::endl;
    std::cout << "Wasserstein distance: " << graph_cdf_diff->Integral() << std::endl;

    return 0;
}