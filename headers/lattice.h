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
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> mx;
    std::vector<float> my;
    std::vector<float> xMainAll;
    std::vector<float> yMainAll;
    std::vector<float> mxMainAll;
    std::vector<float> myMainAll;
    std::vector<float> xAddAll;
    std::vector<float> yAddAll;
    std::vector<float> mxAddAll;
    std::vector<float> myAddAll;
    float *xMainElementaryA = nullptr;
    float *xMainElementaryB = nullptr;
    float *yMainElementaryA = nullptr;
    float *yMainElementaryB = nullptr;
    float *mxMainElementaryA = nullptr;
    float *mxMainElementaryB = nullptr;
    float *myMainElementaryA = nullptr;
    float *myMainElementaryB = nullptr;
    float *xAddElementaryA = nullptr;
    float *xAddElementaryB = nullptr;
    float *yAddElementaryA = nullptr;
    float *yAddElementaryB = nullptr;
    float *mxAddElementaryA = nullptr;
    float *mxAddElementaryB = nullptr;
    float *myAddElementaryA = nullptr;
    float *myAddElementaryB = nullptr;
    float *xMainUnifying = nullptr;
    float *yMainUnifying = nullptr;
    float *mxMainUnifying = nullptr;
    float *myMainUnifying = nullptr;
    float *xAddUnifying = nullptr;
    float *yAddUnifying = nullptr;
    float *mxAddUnifying = nullptr;
    float *myAddUnifying = nullptr;
    std::vector<float> xUnique;
    std::vector<float> xUniqueVector;
    int latticeSize = 0;
    int latticeLinearSize = 0;
    float iteractionRadius = 0;
    float accuracy = 0;
    float splitSeed = 0;
    int layer = 0;
    int layers = 0;
    size_t layerMainSize = 0; //колличество спинов в крайнем слое
    size_t layerMainElementarySize = 0; //колличество спинов взаимодействующих между
                                        //собой в крайнем слое
    size_t layerAddSize = 0; //колличество спинов в присоединяемом слое
    size_t layerAddElementarySize = 0; //колличество спинов взаимодействующих между
                                       //собой в присоединяемом слое
    size_t layerResultSize = 0; //колличество спинов взаимодейстующих
                                //между слоями
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
    void mapMaker();
    virtual void latticeMainMalloc() = 0;
    virtual void latticeAddMalloc() = 0;
    virtual void latticeResultMalloc() = 0;
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
    virtual void latticeMainMalloc();
    virtual void latticeAddMalloc();
    virtual void latticeResultMalloc();
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
    virtual void latticeMainMalloc();
    virtual void latticeAddMalloc();
    virtual void latticeResultMalloc();
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
    virtual void latticeMainMalloc(){};
    virtual void latticeAddMalloc(){};
    virtual void latticeResultMalloc(){};
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
