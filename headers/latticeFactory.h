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
    double *x = nullptr;
    double *y = nullptr;
    double *mx = nullptr;
    double *my = nullptr;
public:
    virtual void createDOS(int linearSize){};
    void read(std::string readPass);
    void generateLattice(int linearSize){};
    virtual void calculate(){};
    void print();
    virtual ~Lattice(){};
};

class LatticeGPU : public Lattice
{
public:
    void createDOS(int linearSize);
    void calculate();
    ~LatticeGPU();
};

class LatticeCPU : public Lattice
{
public:
    void createDOS(int linearSize){};
    void calculate(){};
};

class LatticeGibrid : public Lattice
{
public:
    void createDOS(int linearSize){};
    void calculate(){};
};

class LatticeFactory
{
public:
    static Lattice* createLattice(std::string latticeType);
};

#endif
