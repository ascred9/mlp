#include "include/drawer.h"
#include "include/network.h"

#include "TCanvas.h"
#include "TArrow.h"
#include "TEllipse.h"
#include "TText.h"
#include "TColor.h"
#include "TStyle.h"

void drawNode(const NodePrimitive& n)
{
    TEllipse* circle = new TEllipse(n.m_x, n.m_y, n.m_r);
    circle->Draw();
    TText* text = new TText(n.m_x-n.m_r*sqrt(3.)/2., n.m_y-n.m_r*0.5, Form("%.2f", n.m_val));
    text->SetTextSize(n.m_r);
    text->Draw();
}

void drawConnection(const ConnectionPrimitive& c)
{
    float arrow_size = 0.01;
    int ncolors = TColor::GetNumberOfColors();
    int color = gStyle->GetColorPalette((ncolors-1) * c.m_val);

    float angle = atan2(c.m_y2-c.m_y1, c.m_x2-c.m_x1);
    TArrow* arrow = new TArrow(c.m_x1+c.m_r1*cos(angle), c.m_y1+c.m_r1*sin(angle),
                               c.m_x2-c.m_r2*cos(angle), c.m_y2-c.m_r2*sin(angle),
                               arrow_size, "|>");
    arrow->SetLineColor(color);
    arrow->SetFillColor(color);
    arrow->SetLineWidth(2);
    arrow->Draw();
}

void DrawNet(Network* net)
{
    gStyle->SetPalette(kThermometer);

    TCanvas* c = new TCanvas("cnet", "Neural Network", 1200, 900);
    PrimitiveDrawer drawer(net);
    NodePrimitive::set_draw_func(std::function<void(const NodePrimitive& node)>(drawNode));
    ConnectionPrimitive::set_draw_func(std::function<void(const ConnectionPrimitive& connection)>(drawConnection));
    drawer.draw();
}