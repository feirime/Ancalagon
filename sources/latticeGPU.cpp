#include "lattice.h"

void LatticeGPU::latticeMalloc()
{
    latticeConstructorAdapter(x, y, mx, my, latticeSize);
}

void LatticeGPU::dosMalloc()
{
    latticeConstructorDOSAdapter(Geven, Eeven, Meven, 0);
    latticeConstructorDOSAdapter(Godd, Eodd, Modd, 0);
    latticeConstructorDOSAdapter(Gadd, Eadd, Madd, 0);
}

void LatticeGPU::addCalculate()
{}

void LatticeGPU::calculate()
{}

LatticeGPU::~LatticeGPU()
{
    std::cout << "GPU destructor\n";
    latticeDestructorAdapter(Geven, Eeven, Meven, Godd, Eodd, Modd, x, y, mx, my);
}
