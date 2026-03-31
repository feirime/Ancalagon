#ifndef LATTICEFACTORY_H
#define LATTICEFACTORY_H
#include <fstream>
#include <iostream>
#include <algorithm>
#include <regex>
#include <string>
#include <math.h>
#include "adapterGPU.h"
#include "redAndBlackTree.h"
#include "redAndBlackTreeSE.h"

class Lattice
{
protected:
    unsigned long long *Gresult = nullptr;
    float *Eresult = nullptr;
    float *Mresult = nullptr;
    unsigned long long *confResult = nullptr;
    unsigned long long *Gmain = nullptr;
    float *Emain = nullptr;
    float *Mmain = nullptr;
    unsigned long long *confMain = nullptr;
    unsigned long long *Gadd = nullptr;
    float *Eadd = nullptr;
    float *Madd = nullptr;
    unsigned long long *confAdd = nullptr;
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
    std::vector<float> xUniqueVector;
    int latticeSize = 0;
    int latticeLinearSize = 0;
    float iteractionRadius = 0;
    float accuracy = 0;
    float splitSeed = 0;
    int layer = 0;
    int layers = 0;
    size_t layerResultSize = 0; //колличество спинов в результирующем слое
    size_t layerMainSize = 0; //колличество спинов в крайнем слое
    size_t layerAddSize = 0; //колличество спинов в присоединяемом слое
    size_t dosMainSize = 0;
    size_t dosAddSize = 0;
    size_t dosResultSize = 0;
    size_t confMainSize = 0;
    size_t confAddSize = 0;
    size_t confResultSize = 0;
public:
    int read(std::string readPass);
    void generateLattice(){};
    void init(float iteractionRadius, float accuracy, float splitSeed);
    void layerPlusPlus();
    bool isStart();
    bool isEnd();
    void mapMakerStart();
    void mapMaker();
    virtual void latticeMalloc() = 0;
    virtual void latticeMainMalloc() = 0;
    virtual void latticeAddMalloc() = 0;
    virtual void dosResultMalloc() = 0;
    virtual void dosMainMalloc() = 0;
    virtual void dosAddMalloc() = 0;
    virtual void dosResultFree() = 0;
    virtual void dosMainFree() = 0;
    virtual void dosAddFree() = 0;
    virtual void calculateMain() = 0;
    virtual void calculateAdd() = 0;
    virtual void calculateUnified() = 0;
    virtual void dosMallocBrutforce(){};
    virtual void brutforce();
    void compress();
    void compressRBTree();
    void compressRBTreeSE();
    void compressUMapResult();
    void compressUMapMain();
    void compressUMapAdd();
    void print();
    void print(std::string fileName);
    virtual ~Lattice();
};

class LatticeGPU : public Lattice
{
public:
    void latticeMalloc();
    virtual void latticeMainMalloc();
    virtual void latticeAddMalloc();
    void dosMainMalloc();
    void dosAddMalloc();
    void dosResultMalloc();
    void dosMainFree();
    void dosAddFree();
    void dosResultFree();
    void calculateMain();
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
    void dosMainMalloc();
    void dosAddMalloc();
    void dosResultMalloc();
    void dosMainFree();
    void dosAddFree();
    void dosResultFree();
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
    void dosMainMalloc(){};
    void dosAddMalloc(){};
    void dosResultMalloc(){};
    void dosMainFree(){};
    void dosAddFree(){};
    void dosResultFree(){};
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
