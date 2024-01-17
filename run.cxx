{ 
    gROOT->LoadMacro("build/libmlp.so");
    gROOT->LoadMacro("cos_reg.cpp");
    //gROOT->LoadMacro("RootDrawer.cpp");
    gROOT->ProcessLine("process()");
    gROOT->ProcessLine("DrawResult()");

    //gROOT->LoadMacro("regression_main.cpp");
    //gROOT->LoadMacro("classification_main.cpp");
    //gROOT->ProcessLine("process(\"tph_data.root\")");
}
