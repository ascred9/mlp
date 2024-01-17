#include "include/drawer.h"
#include "include/network.h"

#include "TCanvas.h"
#include "TEllipse.h"
#include "TText.h"

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
    std::cout << "Connection" << std::endl;
    //const ConnectionDrawPrimitive connection = dynamic_cast<const ConnectionDrawPrimitive&>(p);
}

void DrawNet(Network* net)
{
    TCanvas* c = new TCanvas("cnet", "Neural Network", 900, 900);
    PrimitiveDrawer drawer(net);
    NodePrimitive::set_draw_func(std::function<void(const NodePrimitive& node)>(drawNode));
    ConnectionPrimitive::set_draw_func(std::function<void(const ConnectionPrimitive& connection)>(drawConnection));
    drawer.draw();
}