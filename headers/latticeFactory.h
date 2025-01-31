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
    virtual void createDOS(int latticeSize){};
    int read(std::string readPass);
    void generateLattice(int latticeSize){};
    virtual void calculate(int latticeSize){};
    void print();
    virtual ~Lattice(){};
};

class LatticeGPU : public Lattice
{
public:
    void createDOS(int latticeSize);
    void calculate(int latticeSize);
    ~LatticeGPU();
};

class LatticeCPU : public Lattice
{
public:
    void createDOS(int latticeSize){};
    void calculate(int latticeSize){};
};

class LatticeGibrid : public Lattice
{
public:
    void createDOS(int latticeSize){};
    void calculate(int latticeSize){};
};

class LatticeFactory
{
public:
    static Lattice* createLattice(std::string latticeType);
};

#endif
