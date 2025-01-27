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
public:
    virtual void createLattice(){};
    virtual void calculate(){};
    virtual void print(){};
    virtual ~Lattice();
};

class LatticeSquare : public Lattice
{
public:
    void createLattice();
    void calculate();
    void print();
};

class LatticeFactory
{
public:
    static Lattice* createLattice(std::string latticeType);
};

#endif
