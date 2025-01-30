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
    virtual void createLattice(int linearSize){};
    virtual void calculate(){};
    void print();
    virtual ~Lattice(){};
};

class LatticeGPU : public Lattice
{
public:
    void createLattice(int linearSize);
    void calculate();
    ~LatticeGPU();
};

class LatticeCPU : public Lattice
{
public:
    void createLattice(int linearSize){};
    void calculate(){};
};

class LatticeGibrid : public Lattice
{
public:
    void createLattice(int linearSize){};
    void calculate(){};
};

class LatticeFactory
{
public:
    static Lattice* createLattice(std::string latticeType);
};

#endif
