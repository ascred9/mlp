#include <iostream>
#include <set>
#include <numeric>

#include "TString.h"
#include "TFile.h"
#include "TTree.h"


void produce_weights_by_neighbours(TString filename, int N = 100)
{
    TFile* infile = new TFile(filename);
    if (infile->IsZombie())
    {
      std::cout << "file read error" << std::endl;
      return;
    }

    TTree* tph = (TTree*)infile->Get("tph");
    int N_event = tph->GetEntries();

    std::vector<double> weights;
    weights.reserve(N_event);

    std::vector<double> events;
    events.reserve(N_event);

    float simen;
    int fc;
    float en, en0, lxe, csi, bgo, th, phi, rho;
    tph->SetBranchAddress("simen",  &simen);
    tph->SetBranchAddress("fc",     &fc);
    tph->SetBranchAddress("en",     &en);
    tph->SetBranchAddress("en0",    &en0);
    tph->SetBranchAddress("lxe",    &lxe);
    tph->SetBranchAddress("csi",    &csi);
    tph->SetBranchAddress("bgo",    &bgo);
    tph->SetBranchAddress("th",     &th);
    tph->SetBranchAddress("phi",    &phi);
    tph->SetBranchAddress("rho",    &rho);

    std::map<std::pair<int, int>, std::vector<int>> sorted_ev;
    double step_en = 2.;
    double step_th = 0.03;
    for (int iev=0; iev < N_event; ++iev)
    {
        tph->GetEntry(iev);
        int en_ind = simen / step_en;
        int th_ind = abs(th - M_PI/2) / step_th;
        sorted_ev[{en_ind, th_ind}].push_back(iev);
    }

    for (auto it = sorted_ev.begin(); it != sorted_ev.end(); ++it)
        std::cout << it->first.first << " " << it->first.second << " " << it->second.size() << std::endl;

    double D2 = 1.;
    double sigma2_en = 30.;
    double sigma2_th = 0.01;
    for (int iev=0; iev < N_event; ++iev)
    {
        tph->GetEntry(iev);
        if (bgo != 0 || phi > 7 || th > 4 || abs(th-M_PI/2)>0.5) continue;
        double c_sim = simen;
        double c_en = lxe + csi;
        double cn_th = abs(th - M_PI/2);
        double c_rho = rho;
        std::multiset<double> dist_set;

        int en_ind = c_sim / step_en;
        int th_ind = cn_th / step_th;
        std::vector<std::pair<int, int>> v{{-1, -1}, {-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}};
        std::set<int> checks;
        for (auto& sh: v){
            if (sorted_ev.find({en_ind + sh.first, th_ind + sh.second}) == sorted_ev.end())
                continue;
            
            checks.insert(sorted_ev.at({en_ind + sh.first, th_ind + sh.second}).begin(), 
                            sorted_ev.at({en_ind + sh.first, th_ind + sh.second}).end());
        }

        for (int jev: checks)
        {
            tph->GetEntry(jev);

            if (iev == jev || abs(rho - c_rho) > 0.5 || abs(c_sim - simen) > step_en)
                continue;

            double n_th = abs(th - M_PI/2);
            double d2 = pow(c_en - lxe - csi, 2)/sigma2_en + pow(cn_th - n_th, 2)/sigma2_th;
            dist_set.insert(d2);
            
            if (dist_set.size() > N)
                dist_set.erase(std::prev(dist_set.end()), dist_set.end());
        }

        double degree = std::accumulate(dist_set.begin(), dist_set.end(), 0.) / dist_set.size();
	    double weight = degree;
        //double weight = std::exp(-degree / D2);
        std::cout << "event " << iev << ": " << weight << std::endl;
        std::cout << "\t" << c_sim << " " << c_en << " " << cn_th << std::endl;
        std::cout << "\t" << en_ind << " " << th_ind << std::endl;

        events.emplace_back(iev);
        weights.emplace_back(weight);
    }

    //double max = *std::max_element(weights.begin(), weights.end());
    //std::for_each(weights.begin(), weights.end(), [max](double elem) {
    //    return elem/max;});

    TFile* outfile = new TFile("weights.root", "recreate");
    TTree* t = new TTree("weights", "weights");
    int nev;
    float weight;
    t->Branch("nev",    &nev);
    t->Branch("weight", &weight);
    
    if (events.size() != weights.size())
        return;

    for (int i=0; i < events.size(); i++)
    {
        nev = events.at(i);
        weight = weights.at(i);
        t->Fill();
    }
    t->Write();
}

int main()
{
    produce_weights_by_neighbours("tph_data.root");
    return 1;   
}
