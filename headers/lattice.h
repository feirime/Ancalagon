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
    long long int *Gresult = nullptr;
    float *Eresult = nullptr;
    float *Mresult = nullptr;
    long long int *confResult = nullptr;
    long long int *Gmain = nullptr;
    float *Emain = nullptr;
    float *Mmain = nullptr;
    long long int *confMain = nullptr;
    long long int *Gadd = nullptr;
    float *Eadd = nullptr;
    float *Madd = nullptr;
    long long int *confAdd = nullptr;
    float *x = nullptr;
    float *y = nullptr;
    float *mx = nullptr;
    float *my = nullptr;
    float *xMain = nullptr;
    float *yMain = nullptr;
    float *mxMain = nullptr;
    float *myMain = nullptr;
    float *xAdd = nullptr;
    float *yAdd = nullptr;
    float *mxAdd = nullptr;
    float *myAdd = nullptr;
    int latticeSize = 0;
    int latticeLinearSize = 0;
    float iteractionRadius = 0;
    float splitSeed = 0;
    int layer = 0;
    int layers = 0;
    unsigned long long int confs = 0;
    size_t layerMainSize = 0; //колличество спинов в крайнем слое
    size_t layerResultSize = 0; //колличество спинов в результирующем слое
    size_t layerAddSize = 0; //колличество спинов в присоединяемом слое
public:
    int read(std::string readPass);
    void generateLattice(){};
    void splitInit(float iteractionRadius, float splitSeed);
    void layerPlusPlus();
    void compress();
    void mapMaker();
    virtual void latticeMalloc() = 0;
    virtual void latticeMainMalloc() = 0;
    virtual void latticeAddMalloc() = 0;
    virtual void dosCopyMalloc() = 0;
    virtual void calculateMain() = 0;
    virtual void calculateAdd() = 0;
    virtual void calculateUnified() = 0;
    virtual void dosMallocBrutforce(){};
    virtual void brutforce();
    void print();
    bool isStart();
    bool isEnd();
    virtual ~Lattice();
};

class LatticeGPU : public Lattice
{
public:
    void latticeMalloc();
    virtual void latticeMainMalloc(){};
    virtual void latticeAddMalloc(){};
    void dosCopyMalloc();
    void calculateMain(){};
    void calculateAdd();
    void calculateUnified();
    ~LatticeGPU();
};

class LatticeCPU : public Lattice
{
public:
    void latticeMalloc();
    virtual void latticeMainMalloc();
    virtual void latticeAddMalloc();
    void dosCopyMalloc();
    void calculateMain();
    void calculateAdd();
    void calculateUnified();
    void dosMallocBrutforce();
    ~LatticeCPU();
};

class LatticeGibrid : public Lattice
{
public:
    void latticeMalloc(){};
    virtual void latticeMainMalloc(){};
    virtual void latticeAddMalloc(){};
    void dosCopyMalloc(){};
    void calculateMain(){};
    void calculateAdd(){};
    void calculateUnified(){};
    ~LatticeGibrid(){};
};

class LatticeFactory
{
public:
    static Lattice* createLattice(std::string latticeType);
};

#endif
