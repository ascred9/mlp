{ 
    gROOT->LoadMacro("build/libmlp.so");
    gROOT->LoadMacro("kde_test.cpp");
    gROOT->ProcessLine("test_kde()");
}
