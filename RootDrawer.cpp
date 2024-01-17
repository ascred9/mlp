#include "include/drawer.h"
#include "include/network.h"

#include "TCanvas.h"

void drawNode(const Primitive& p)
{
    std::cout << "TypeOf" << typeid(p).name() << " Node" << std::endl;
    //const NodeDrawPrimitive node = dynamic_cast<const NodeDrawPrimitive&>(p);
    //std::cout << node.radii << std::endl;
}

void drawConnection(const Primitive& p)
{
    std::cout << "TypeOf" << typeid(p).name() << " Connection" << std::endl;
    //const ConnectionDrawPrimitive connection = dynamic_cast<const ConnectionDrawPrimitive&>(p);
}

void DrawNet(Network* net)
{
    TCanvas* c = new TCanvas("cnet", "Neural Network", 900, 900);
    PrimitiveDrawer drawer(net);
    NodeDrawPrimitive::set_draw_func(std::function<void(const Primitive& node)>(drawNode));
    ConnectionDrawPrimitive::set_draw_func(std::function<void(const Primitive& connection)>(drawConnection));
    drawer.draw();
}