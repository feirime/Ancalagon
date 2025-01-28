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
    int linearSize;
    long long int lineaarLength;
    std::string device;
    bool lattice_read;
    std::string read_pass;
    std::string latticeType;
    LatticeFactory *lattice;
public:
    void run(int argc, char* argv[]);
    void arguments(int argc, char* argv[]);
};

#endif
