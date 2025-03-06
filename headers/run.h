#ifndef RUN_H
#define RUN_H

#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <chrono>
#include <argumentum/argparse.h>
#include "adapterGPU.h"
#include "latticeFactory.h"

class Run
{
private:
    int latticeSize;
    float splitSeed;
    long long int lineaarLength;
    std::string device;
    bool lattice_read;
    std::string readPass;
    std::string *fileName = nullptr;
    int numberOfFiles;
    //std::string latticeType;
    LatticeFactory *lattice;
public:
    void run(int argc, char* argv[]);
    void readIterator();
    void arguments(int argc, char* argv[]);
    ~Run();
};

#endif
