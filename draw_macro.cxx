#include "TCanvas.h"
#include "TFile.h"
#include "TF1.h"
#include "TPad.h"
#include "TTree.h"

#include "include/kde.h"

#include <iostream>

Double_t LogGaus(Double_t* x, Double_t *p)
{
    Double_t xi = 2 * sqrt(2*log(2));
    Double_t s0 = 2/xi * log(xi*p[3]/2 + sqrt(1 + pow(xi*p[3]/2, 2)));
    return p[0] * p[3]/(sqrt(2*M_PI) * p[2] * s0) * exp(-pow(log(1 - p[3] * (x[0]-p[1]) / p[2])/s0, 2)/2 - pow(s0, 2)/2);
}

void draw_macro(TString filename = "out.root")
{
    bool flag = false;
    //bool flag = true;

    TFile* infile = new TFile(filename);
    if (infile->IsZombie())
    {
        std::cout << "file read error" << std::endl;
        return;
    }
    
    TTree* tph = (TTree*)infile->Get("tph");
    float simen;
    tph->SetBranchAddress("simen", &simen);
    TTree* tnet = (TTree*)infile->Get("tnet");


    TCanvas* c1 = new TCanvas("neuron_net", "neuron net results", 900, 600);
    c1->Divide(2, 2);

    c1->cd(1)->SetGrid();
    c1->cd(1);
    if (flag)
        tph->Draw("(lxe+csi)/rec - simen:simen", "abs((lxe+csi)/rec - simen)<150");
    else
        tph->Draw("rec - simen:simen", "abs(rec - simen)<150", "");
    
    
    //c1->cd(2)->Divide(2, 1);
    c1->cd(2)->SetGrid();//->cd(1);
    c1->cd(2);//->cd(1);
    if (flag)
        tph->Draw("(lxe+csi)/rec - simen>>h2(400)", "abs((lxe+csi)/rec - simen)<150");
    else
        tph->Draw("rec - simen>>h2(400)", "abs(rec - simen)<150");

    tph->Draw("en - simen>>h2origin(400)", "abs(en - simen)<150", "same");
    gROOT->ProcessLine("h2->SetLineColor(kBlack)");
    TF1* logGaus = new TF1("logGaus", LogGaus, -60, 60, 4);
    logGaus->SetParNames("A", "m", "s", "eta");
    logGaus->SetParameter(0, 1000);
    logGaus->SetParameter(1, 0);
    logGaus->SetParameter(2, 30);
    logGaus->SetParameter(3, 0.1);
    gROOT->ProcessLine("h2->Fit(\"logGaus\", \"\", \"\", -50, 30)");
    gROOT->ProcessLine("h2origin->Fit(\"logGaus\", \"\", \"\", -50, 20)");

    //c1->cd(2)->cd(2);
    //if (flag)
    //    tph->Draw("pow((lxe+csi)/rec - simen,2):simen", "abs((lxe+csi)/rec - simen)<150");
    //else
    //    tph->Draw("pow(rec - simen,2):simen", "abs(rec - simen)<150");
    
    c1->cd(3)->Divide(1, 2);
    c1->cd(3)->cd(1)->SetGrid();
    c1->cd(3)->cd(1);
    if (flag)
        tph->Draw("(lxe+csi)/rec - simen:abs(th-3.1415/2)", "abs((lxe+csi)/rec - simen)<150");
    else
        tph->Draw("rec - simen:abs(th-3.1415/2)", "abs(rec - simen)<150");

    c1->cd(3)->cd(2)->SetGrid();
    c1->cd(3)->cd(2);
    tnet->Draw("mean_loss:nepoch", "nepoch>2", "L");

    c1->cd(4)->Divide(1, 2);
    c1->cd(4)->cd(1)->SetGrid();
    c1->cd(4)->cd(1);
    TPad* pad = new TPad("pad4", "pad4", 0, 0, 1, 1);
    pad->Draw();
    pad->cd();
    pad->SetLogz();
    if (flag)
        tph->Draw("(lxe+csi)/rec - simen:rho>>h4(30, 30, 200, 200)", "abs((lxe+csi)/rec - simen)<100", "lego");
    else
        tph->Draw("rec - simen:rho>>h4(30, 30, 200, 200)", "abs(rec - simen)<100", "lego");

    c1->cd(4)->cd(2);
    TCanvas* c2 = new TCanvas();
    c2->cd()->SetGrid();
    c2->cd();
    if (flag)
    {
        tph->Draw("(lxe+csi)/rec>>h5(700, 800, 100, 100)");
    }
    else
    {
        tph->Draw("rec>>h5(300, 0, 300)");
    }
    gROOT->ProcessLine("h5->SetLineColor(kRed)");
    tph->Draw("simen", "", "same");
    tph->Draw("en", "", "same");

    double sigma = 0.2 * 50;
    double xi = 2 * sqrt(2*log(2));
    double eta = 1.5e-01;
    double s0 = 2/xi * log(xi*eta/2 + sqrt(1 + pow(xi*eta/2, 2)));
    //auto expected_f = [sigma](double* xx, double* par){
    //    double x = xx[0] - 150;
    //    return 0.25 * par[0] * (std::erf((50-x)/(sqrt(2)*sigma)) + std::erf((50+x)/(sqrt(2)*sigma)));};
    //auto expected_f = [sigma, eta, s0](double* xx, double* par){
    auto expected_f = [sigma, eta, s0](double* xx, double* par){
                                              double x = xx[0] - 150;
                                              double eta = par[2];
                                              double sigma = par[1];
                                              double xi = 2 * sqrt(2*log(2));
                                              double s0 = 2/xi * log(xi*eta/2 + sqrt(1 + pow(xi*eta/2, 2)));
                                              double part1 = 1., part2 = 1.;
                                              if ((x+50)*eta/sigma < 1)
                                                part1 = std::erf( (s0*s0 - log(1 - (x+50)*eta/sigma)) / (sqrt(2) * s0) );
                                              if((x-50)*eta/sigma < 1)
                                                part2 = std::erf( (s0*s0 - log(1 - (x-50)*eta/sigma)) / (sqrt(2) * s0) );
                                              return 0.25 * par[0] * (part1 - part2);};
    TF1* f = new TF1("fexpected", expected_f, 50, 250, 3);
    f->SetParameter(0, 2000);
    f->SetParameter(1, 10);
    f->SetParameter(2, 0.15);
    gROOT->ProcessLine("h5->Fit(\"fexpected\")");
    f->SetLineStyle(kDashed);
    f->SetLineColor(kBlack);
    f->Draw("same");
    
    TH1D* hist_random = new TH1D("hist_random", "hist_random", 300, 0, 300);
    NovosibirskGenerator gen(0, 12.71, 0.1747);
    for (int i = 0; i < tph->GetEntries(); i++)
    {
        tph->GetEntry(i);
        hist_random->Fill(simen + gen.generate());
    }
    hist_random->SetLineColor(kGreen);
    hist_random->Draw("same");

    if (false)
    {
        TCanvas* c2 = new TCanvas("neuron_net2", "neuron net results", 900, 600);
        c2->Divide(2, 3);
        c2->cd(1);
        tph->Draw("rr_lxe:lxe");
        c2->cd(2);
        tph->Draw("rr_lxe-lxe");
        c2->cd(3);
        tph->Draw("rr_csi:csi");
        c2->cd(4);
        tph->Draw("rr_csi-csi");
        c2->cd(5);
        tph->Draw("csi:lxe>>h3");
        gROOT->ProcessLine("h3->SetMarkerColor(kRed)");
        gROOT->ProcessLine("h3->Draw()");
        tph->Draw("rr_csi:rr_lxe", "", "same");
        c2->cd(6);
        tph->Draw("rr_csi-csi:rr_lxe-lxe");
    }
}
