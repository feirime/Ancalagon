#include "lattice.h"

void LatticeGPU::latticeMainMalloc()
{
    latticeConstructorAdapter(xMainElementaryA, yMainElementaryA, mxMainElementaryA, myMainElementaryA, layerMainElementarySize);
    latticeConstructorAdapter(xMainElementaryB, yMainElementaryB, mxMainElementaryB, myMainElementaryB, layerMainElementarySize);
}

void LatticeGPU::latticeAddMalloc()
{
    latticeConstructorAdapter(xAddElementaryA, yAddElementaryA, mxAddElementaryA, myAddElementaryA, layerAddElementarySize);
    latticeConstructorAdapter(xAddElementaryB, yAddElementaryB, mxAddElementaryB, myAddElementaryB, layerAddElementarySize);
}

void LatticeGPU::latticeResultMalloc()
{
    latticeConstructorAdapter(xMainUnifying, yMainUnifying, mxMainUnifying, myMainUnifying, layerResultSize);
}

void LatticeGPU::dosMainMalloc()
{
    dosConstructorAdapter(Gmain, Emain, Mmain, confMain, dosMainSize);
}

void LatticeGPU::dosAddMalloc()
{
    dosConstructorAdapter(Gadd, Eadd, Madd, confAdd, dosAddSize);
}

void LatticeGPU::dosResultMalloc()
{
    dosConstructorAdapter(Gresult, Eresult, Mresult, confResult, dosResultSize);
}

void LatticeGPU::dosMainFree()
{
    dosDestructorAdapter(Gmain, Emain, Mmain, confMain);
}

void LatticeGPU::dosAddFree()
{
    dosDestructorAdapter(Gadd, Eadd, Madd, confAdd);
}

void LatticeGPU::dosResultFree()
{
    dosDestructorAdapter(Gresult, Eresult, Mresult, confResult);
}

void LatticeGPU::calculateMain()
{
    kernelElementaryAdapter(xMain, yMain, mxMain, myMain, layerMainSize, Gmain, Emain, Mmain, confMain, 
        dosMainSize, confMainSize, iteractionRadius);
}

void LatticeGPU::calculateAdd()
{
    kernelElementaryAdapter(xAdd, yAdd, mxAdd, myAdd, layerAddSize, Gadd, Eadd, Madd, confAdd, 
        dosAddSize, confAddSize, iteractionRadius);
}

void LatticeGPU::calculateUnified()
{
    kernelUnifyingAdapter(xMain, yMain, mxMain, myMain, layerMainSize,
        xAdd, yAdd, mxAdd, myAdd, layerAddSize,
        Gmain, Emain, Mmain, confMain, dosMainSize, confMainSize,
        Gadd, Eadd, Madd, confAdd, dosAddSize, confAddSize,
        Gresult, Eresult, Mresult, confResult, dosResultSize, confResultSize,
        iteractionRadius);
}

LatticeGPU::~LatticeGPU()
{
    //std::cout << "GPU destructor\n";
    latticeDestructorAdapter(xMain, yMain, mxMain, myMain);
    latticeDestructorAdapter(xAdd, yAdd, mxAdd, myAdd);
    dosDestructorAdapter(Gmain, Emain, Mmain, confMain);
    dosDestructorAdapter(Gadd, Eadd, Madd, confAdd);
    dosDestructorAdapter(Gresult, Eresult, Mresult, confResult);
}
