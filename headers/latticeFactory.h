#ifndef LATTICEFACTORY_H
#define LATTICEFACTORY_H

#include <string>
#include "adapterGPU.h"

class Lattice
{
protected:
    long long int *G = nullptr;
    float *E = nullptr;
    int *M = nullptr;
    float *x = nullptr;
    float *y = nullptr;
    float *mx = nullptr;
    float *my = nullptr;
public:
    virtual void createDOS(int latticeSize){};
    int read(std::string readPass);
    void generateLattice(int latticeSize){};
    virtual void calculate(int latticeSize, float splitSeed){};
    void print();
    virtual ~Lattice(){};
};

class LatticeGPU : public Lattice
{
public:
    void createDOS(int latticeSize);
    void calculate(int latticeSize, float splitSeed);
    ~LatticeGPU();
};

class LatticeCPU : public Lattice
{
public:
    void createDOS(int latticeSize){};
    void calculate(int latticeSize, float splitSeed){};
};

class LatticeGibrid : public Lattice
{
public:
    void createDOS(int latticeSize){};
    void calculate(int latticeSize, float splitSeed){};
};

class LatticeFactory
{
public:
    static Lattice* createLattice(std::string latticeType);
};

#endif
