#include "lattice.h"

void LatticeGPU::latticeMalloc()
{
    latticeConstructorAdapter(x, y, mx, my, latticeSize);
}

void LatticeGPU::createDOS() 
{
    latticeConstructorDOSAdapter(G, E, M);
    latticeConstructorAdapter(x, y, mx, my, latticeSize);
};

void LatticeGPU::calculate(float splitSeed)
{
    if(E != nullptr && G != nullptr && M != nullptr)
    {
        calculateAdapter(G, E, M, x, y, mx, my, latticeSize, splitSeed);
    }
}

LatticeGPU::~LatticeGPU() 
{
    std::cout << "GPU destructor\n";
    latticeDestructorAdapter(G, E, M, x, y, mx, my);
}
