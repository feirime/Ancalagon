#ifndef LATTICEFACTORY_H
#define LATTICEFACTORY_H
#include <fstream>
#include <iostream>
#include <algorithm>
#include <regex>
#include <string>
#include <math.h>
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
    float *mainLayerX = nullptr;
    float *mainLayerY = nullptr;
    float *mainLayerMx = nullptr;
    float *mainLayerMy = nullptr;
    float *connectedSpinsX = nullptr;
    float *connectedSpinsY = nullptr;
    float *connectedSpinsMx = nullptr;
    float *connectedSpinsMy = nullptr;
    int latticeSize = 0;
    float iteractionRadius = 0;
    float splitSeed = 0;
public:
    int read(std::string readPass);
    void generateLattice(){};
    void initializeLattice(float iteractionRadius, float splitSeed);
    virtual void latticeMalloc(){};
    virtual void dosMalloc(){};
    virtual void calculate(){};
    void print();
    virtual ~Lattice();
};

class LatticeGPU : public Lattice
{
public:
    void latticeMalloc();
    void dosMalloc();
    void calculate();
    ~LatticeGPU();
};

class LatticeCPU : public Lattice
{
public:
    void latticeMalloc();
    void dosMalloc();
    unsigned int mainMapMaker();
    unsigned int connectedMapMaker();
    void calculate();
    ~LatticeCPU();
};

class LatticeGibrid : public Lattice
{
public:
    void dosMalloc(){};
    void calculate(){};
};

class LatticeFactory
{
public:
    static Lattice* createLattice(std::string latticeType);
};

#endif
