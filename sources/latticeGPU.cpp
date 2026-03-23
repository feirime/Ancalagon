#include "lattice.h"

void LatticeGPU::latticeMalloc()
{
    latticeConstructorAdapter(x, y, mx, my, latticeSize);
}

void LatticeGPU::latticeMainMalloc()
{
    latticeMainConstructorAdapter(xMain, yMain, mxMain, myMain, layerMainSize);
}

void LatticeGPU::latticeAddMalloc()
{
    latticeAddConstructorAdapter(xAdd, yAdd, mxAdd, myAdd, layerAddSize);
}

void LatticeGPU::dosMainMalloc()
{
    latticeConstructorDOSAdapter(Gmain, Emain, Mmain, dosMainSize);
}

void LatticeGPU::dosAddMalloc()
{
    latticeConstructorDOSAdapter(Gadd, Eadd, Madd, dosAddSize);
}

void LatticeGPU::dosResultMalloc()
{
    latticeConstructorDOSAdapter(Gresult, Eresult, Mresult, dosMainSize);
}

void LatticeGPU::dosMainFree()
{
    latticeDestructorDOSAdapter(Gmain, Emain, Mmain);
}

void LatticeGPU::dosAddFree()
{
    latticeDestructorDOSAdapter(Gadd, Eadd, Madd);
}

void LatticeGPU::dosResultFree()
{
    latticeDestructorDOSAdapter(Gresult, Eresult, Mresult);
}

void LatticeGPU::calculateMain()
{}

void LatticeGPU::calculateAdd()
{}

void LatticeGPU::calculateUnified()
{}

LatticeGPU::~LatticeGPU()
{
    //std::cout << "GPU destructor\n";
    latticeAllDestructorAdapter(Gmain, Emain, Mmain, Gresult, Eresult, Mresult, x, y, mx, my, 
        xMain, yMain, mxMain, myMain, xAdd, yAdd, mxAdd, myAdd);
}
