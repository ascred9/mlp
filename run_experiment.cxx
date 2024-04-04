{ 
    gROOT->LoadMacro("build/libmlp.so");

    gROOT->LoadMacro("regression_experiment.cpp");
    gROOT->ProcessLine("process(\"2gamma.root\")");
    gROOT->LoadMacro("draw_macro.cxx");
    gROOT->ProcessLine("draw_macro(\"out_experiment.root\")");
}
