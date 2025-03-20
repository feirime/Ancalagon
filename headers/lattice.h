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
    int latticeSize = 0;
public:
    int read(std::string readPass);
    void generateLattice(){};
    virtual void latticeMalloc(){};
    virtual void createDOS(){};
    virtual void calculate(float splitSeed){};
    void print();
    virtual ~Lattice();
};

class LatticeGPU : public Lattice
{
public:
    void latticeMalloc();
    void createDOS();
    void calculate(float splitSeed);
    ~LatticeGPU();
};

class LatticeCPU : public Lattice
{
public:
    void latticeMalloc();
    void createDOS();
    void mainMapMaker();
    void connectedMapMaker();
    void calculate(float splitSeed);
    ~LatticeCPU();
};

class LatticeGibrid : public Lattice
{
public:
    void createDOS(){};
    void calculate(float splitSeed){};
};

class LatticeFactory
{
public:
    static Lattice* createLattice(std::string latticeType);
};

#endif
