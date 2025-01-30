#include "latticeFactory.h"

Lattice* LatticeFactory::createLattice(std::string device) 
{
    if (device == "GPU") 
    {
        return new LatticeGPU();
    }
    else if (device == "CPU") 
    {
        return new LatticeCPU();
    }
    else if (device == "gibrid") 
    {
        return new LatticeGibrid();
    }
    else 
    {
        std::exit(1);
    }
}

LatticeGPU::~LatticeGPU() 
{
    latticeDestructorAdapter(G, E, M, x, y, mx, my);
}

void LatticeGPU::createLattice(int linearSize) 
{
    latticeConstructorAdapter(G, E, M, x, y, mx, my, linearSize);
};

void LatticeGPU::calculate()
{
    if(E != nullptr && G != nullptr && M != nullptr)
    {
        calculateAdapter(G, E, M);
    }
}

void Lattice::print()
{}
