#include "TCanvas.h"
#include "TFile.h"
#include "TF1.h"
#include "TPad.h"
#include "TTree.h"

#include <iostream>

void draw_macro()
{
    TFile* infile = new TFile("out.root");
    if (infile->IsZombie())
    {
        std::cout << "file read error" << std::endl;
        return;
    }
    
    TTree* tph = (TTree*)infile->Get("tph");


    TCanvas* c1 = new TCanvas("neuron_net", "neuron net results", 900, 600);
    c1->Divide(2, 2);

    c1->cd(1);
    tph->Draw("(lxe+csi)/rec - simen:simen", "abs((lxe+csi)/rec - simen)<150");
    
    
    c1->cd(2);
    tph->Draw("(lxe+csi)/rec - simen>>h2(400)", "abs((lxe+csi)/rec - simen)<150");
    gROOT->ProcessLine("h2->Fit(\"gaus\", \"\", \"\", -50, 50)");
    
    c1->cd(3);
    tph->Draw("(lxe+csi)/rec - simen:abs(th-3.1415/2)", "abs((lxe+csi)/rec - simen)<150");

    c1->cd(4);
    TPad* pad = new TPad("pad4", "pad4", 0, 0, 1, 1);
    pad->Draw();
    pad->cd();
    pad->SetLogz();
    tph->Draw("(lxe+csi)/rec - simen:rho>>h4(30, 30, 200, 200)", "abs((lxe+csi)/rec - simen)<100", "lego");
}