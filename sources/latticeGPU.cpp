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

void LatticeGPU::dosMalloc()
{
    latticeConstructorDOSAdapter(Gmain, Emain, Mmain, 0);
    latticeConstructorDOSAdapter(Gresult, Eresult, Mresult, 0);
    latticeConstructorDOSAdapter(Gadd, Eadd, Madd, 0);
}

void LatticeGPU::calculateAdd()
{}

void LatticeGPU::calculateUnified()
{}

LatticeGPU::~LatticeGPU()
{
    //std::cout << "GPU destructor\n";
    latticeDestructorAdapter(Gmain, Emain, Mmain, Gresult, Eresult, Mresult, x, y, mx, my);
}
