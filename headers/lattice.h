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
    float *xEven = nullptr;
    float *yEven = nullptr;
    float *mxEven = nullptr;
    float *myEven = nullptr;
    float *xOdd = nullptr;
    float *yOdd = nullptr;
    float *mxOdd = nullptr;
    float *myOdd = nullptr;
    int latticeSize = 0;
    int latticeLinearSize = 0;
    float iteractionRadius = 0;
    float splitSeed = 0;
    int layer = 0;
    int layers = 0;
    size_t layerMainSize = 0; //колличество спинов в крайнем слое
    size_t layerResultSize = 0; //колличество спинов в результирующем слое
    size_t layerConnectedSize = 0; //колличество спинов в присоединяемом слое
public:
    int read(std::string readPass);
    void generateLattice(){};
    void initializeLattice(float iteractionRadius, float splitSeed);
    void addConfigure();
    void compress();
    unsigned int mapMakerEven();
    unsigned int mapMakerOdd();
    virtual void latticeMalloc() = 0;
    virtual void latticeEvenMalloc() = 0;
    virtual void latticeOddMalloc() = 0;
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
    virtual void latticeEvenMalloc(){};
    virtual void latticeOddMalloc(){};
    void dosMalloc();
    void calculate();
    void addCalculate();
    ~LatticeGPU();
};

class LatticeCPU : public Lattice
{
public:
    void latticeMalloc();
    virtual void latticeEvenMalloc();
    virtual void latticeOddMalloc();
    void dosMalloc();
    void addCalculate();
    void calculate();
    ~LatticeCPU();
};

class LatticeGibrid : public Lattice
{
public:
    void latticeMalloc(){};
    virtual void latticeEvenMalloc(){};
    virtual void latticeOddMalloc(){};
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
