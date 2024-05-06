#include "TCanvas.h"
#include "TFile.h"
#include "TF1.h"
#include "TPad.h"
#include "TTree.h"

#include <iostream>

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
    TTree* tnet = (TTree*)infile->Get("tnet");


    TCanvas* c1 = new TCanvas("neuron_net", "neuron net results", 900, 600);
    c1->Divide(2, 2);

    c1->cd(1);
    if (flag)
        tph->Draw("(lxe+csi)/rec - simen:simen", "abs((lxe+csi)/rec - simen)<150");
    else
        tph->Draw("rec - simen:simen", "abs(rec - simen)<150");
    
    
    c1->cd(2)->Divide(2, 1);
    c1->cd(2)->cd(1);
    if (flag)
        tph->Draw("(lxe+csi)/rec - simen>>h2(400)", "abs((lxe+csi)/rec - simen)<150");
    else
        tph->Draw("rec - simen>>h2(400)", "abs(rec - simen)<150");

    tph->Draw("en - simen>>h2origin(400)", "abs(en - simen)<150", "same");
    gROOT->ProcessLine("h2->SetLineColor(kBlack)");
    gROOT->ProcessLine("h2->Fit(\"gausn\", \"\", \"\", -10, 10)");
    gROOT->ProcessLine("h2origin->Fit(\"gausn\", \"\", \"\", -10, 10)");

    c1->cd(2)->cd(2);
    if (flag)
        tph->Draw("pow((lxe+csi)/rec - simen,2):simen", "abs((lxe+csi)/rec - simen)<150");
    else
        tph->Draw("pow(rec - simen,2):simen", "abs(rec - simen)<150");
    
    c1->cd(3)->Divide(1, 2);
    c1->cd(3)->cd(1);
    if (flag)
        tph->Draw("(lxe+csi)/rec - simen:abs(th-3.1415/2)", "abs((lxe+csi)/rec - simen)<150");
    else
        tph->Draw("rec - simen:abs(th-3.1415/2)", "abs(rec - simen)<150");

    c1->cd(3)->cd(2);
    tnet->Draw("mean_loss:nepoch", "nepoch>2", "L");

    c1->cd(4)->Divide(1, 2);
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
    if (flag)
    {
        tph->Draw("(lxe+csi)/rec>>h5(700, 800, 100, 100)");
    }
    else
    {
        tph->Draw("rec>>h5(100, 200, 100, 100)");
    }
    tph->Draw("simen", "", "same");
    
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