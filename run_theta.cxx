{ 
    gROOT->LoadMacro("build/libmlp.so");

    gROOT->LoadMacro("regression_main_large_theta.cpp");
    //gROOT->LoadMacro("classification_main.cpp");
    gROOT->ProcessLine("process(\"tph_data.root\")");
}
