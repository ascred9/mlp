#include "TCanvas.h"
#include "TArrow.h"
#include "TEllipse.h"

TCanvas* Draw()
{
    TCanvas* c1 = new TCanvas("c1", "Neural Network", 900, 900);
    TEllipse* el = new TEllipse(0.65, 0.2, 0.2, 0.1);
    el->SetFillStyle(3000);
    el->Draw();
    return c1;
}