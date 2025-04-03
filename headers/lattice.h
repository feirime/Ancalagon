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
    long long int *Geven = nullptr;
    float *Eeven = nullptr;
    int *Meven = nullptr;
    long long int *Godd = nullptr;
    float *Eodd = nullptr;
    int *Modd = nullptr;
    long long int *Gadd = nullptr;
    float *Eadd = nullptr;
    int *Madd = nullptr;
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
    int latticeLinearSize = 0;
    float iteractionRadius = 0;
    float splitSeed = 0;
    int layer = 0;
    int layers = 0;
    long long int mainLayerSize = 0;
    long long int resultLayerSize = 0;
    long long int connectedSpinsSize = 0;
public:
    int read(std::string readPass);
    void generateLattice(){};
    void initializeLattice(float iteractionRadius, float splitSeed);
    void addConfigure();
    void compress();
    unsigned int mainMapMaker();
    unsigned int connectedMapMaker();
    virtual void latticeMalloc() = 0;
    virtual void dosMalloc() = 0;
    virtual void addCalculate() = 0;
    virtual void calculate() = 0;
    void print();
    bool isEnd();
    virtual ~Lattice();
};

class LatticeGPU : public Lattice
{
public:
    void latticeMalloc();
    void dosMalloc();
    void calculate();
    void addCalculate();
    ~LatticeGPU();
};

class LatticeCPU : public Lattice
{
public:
    void latticeMalloc();
    void dosMalloc();
    void addCalculate();
    void calculate();
    ~LatticeCPU();
};

class LatticeGibrid : public Lattice
{
public:
    void latticeMalloc(){};
    void dosMalloc(){};
    void addCalculate(){};
    void calculate(){};
    ~LatticeGibrid(){};
};

class LatticeFactory
{
public:
    static Lattice* createLattice(std::string latticeType);
};

#endif
