#ifndef LATTICEFACTORY_H
#define LATTICEFACTORY_H

#include <string>
#include "adapterGPU.h"


class Lattice
{
public:
    virtual void createLattice(){};
    virtual ~Lattice(){};
};

class LatticeSquare : public Lattice
{
private:
    long long int *G;
    float *E;
    int *M;
public:
    void createLattice();
    ~LatticeSquare();
};

class LatticeFactory
{
public:
    static Lattice* createLattice(std::string latticeType)
    {
        if (latticeType == "square") 
        {
            return new LatticeSquare();
        }
        else 
        {
            return nullptr;
        }
    }
};

#endif
