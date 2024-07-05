{ 
    gROOT->LoadMacro("build/libmlp.so");
    //gROOT->LoadMacro("cos_reg.cpp");
    //gROOT->LoadMacro("RootDrawer.cpp");
    //gROOT->ProcessLine("process()");
    //gROOT->ProcessLine("DrawResult()");

    gROOT->LoadMacro("regression_main.cpp");
    //gROOT->LoadMacro("classification_main.cpp");
    //gROOT->LoadMacro("encoder.cpp");
    //gROOT->ProcessLine("process(\"tph_data.root\")");
    gROOT->ProcessLine("process(\"tph_100_200.root\")");
    //gROOT->ProcessLine("process(\"tph_500_1000.root\")");
    gROOT->LoadMacro("draw_macro.cxx");
    gROOT->ProcessLine("draw_macro()");
}
