#ifndef RUN_H
#define RUN_H

#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <chrono>
#include <memory>
#include <argumentum/argparse.h>
#include "adapterGPU.h"
#include "lattice.h"

class CalcStrategy
{
public:
    virtual void calculate(Lattice *lattice) = 0;
};

class Brutforce : public CalcStrategy
{
public:
    void calculate(Lattice *lattice) override;
};

class Decomposition : public CalcStrategy
{
public:
    void calculate(Lattice *lattice) override;
};

class Run
{
private:
    int latticeSize=0;
    float splitSeed;
    float iteractionRadius;
    float accuracy;
    std::string device;
    bool lattice_read;
    std::string readPass;
    std::string *fileName = nullptr;
    int numberOfFiles;
    std::string calcStrategyStr;
    std::unique_ptr<CalcStrategy> calcStrategy;
    //std::string latticeType;
    LatticeFactory *lattice;
public:
    void run(int argc, char* argv[]);
    void arguments(int argc, char* argv[]);
    void readIterator();
    void setCalcStrategy();
    void showTime(std::chrono::high_resolution_clock::time_point start, 
        std::chrono::high_resolution_clock::time_point end);
    ~Run();
};

#endif
