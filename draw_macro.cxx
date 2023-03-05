#include "TCanvas.h"
#include "TFile.h"
#include "TF1.h"
#include "TPad.h"
#include "TTree.h"

#include <iostream>

void draw_macro()
{
    bool flag = false;

    TFile* infile = new TFile("out.root");
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
    
    
    c1->cd(2);
    if (flag)
        tph->Draw("(lxe+csi)/rec - simen>>h2(400)", "abs((lxe+csi)/rec - simen)<150");
    else
        tph->Draw("rec - simen>>h2(400)", "abs(rec - simen)<150");

    tph->Draw("en - simen>>h2origin(400)", "abs(en - simen)<150", "same");
    gROOT->ProcessLine("h2->SetLineColor(kBlack)");
    gROOT->ProcessLine("h2->Fit(\"gausn\", \"\", \"\", -50, 50)");
    gROOT->ProcessLine("h2origin->Fit(\"gausn\", \"\", \"\", -50, 50)");
    
    c1->cd(3)->Divide(1, 2);
    c1->cd(3)->cd(1);
    if (flag)
        tph->Draw("(lxe+csi)/rec - simen:abs(th-3.1415/2)", "abs((lxe+csi)/rec - simen)<150");
    else
        tph->Draw("rec - simen:abs(th-3.1415/2)", "abs(rec - simen)<150");

    c1->cd(3)->cd(2);
    tnet->Draw("mean_loss:nepoch", "nepoch>2", "L");

    c1->cd(4);
    TPad* pad = new TPad("pad4", "pad4", 0, 0, 1, 1);
    pad->Draw();
    pad->cd();
    pad->SetLogz();
    if (flag)
        tph->Draw("(lxe+csi)/rec - simen:rho>>h4(30, 30, 200, 200)", "abs((lxe+csi)/rec - simen)<100", "lego");
    else
        tph->Draw("rec - simen:rho>>h4(30, 30, 200, 200)", "abs(rec - simen)<100", "lego");
}