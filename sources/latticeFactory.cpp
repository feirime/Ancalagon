#include "latticeFactory.h"

Lattice* LatticeFactory::createLattice(std::string latticeType) 
{
    if (latticeType == "square") 
    {
        return new LatticeSquare();
    }
    else 
    {
        return nullptr;
    }
}

Lattice::~Lattice() 
{
    latticeDestructorAdapter(G, E, M);
}

void LatticeSquare::createLattice() 
{
    latticeConstructorAdapter(G, E, M);
};

void LatticeSquare::calculate()
{
    if(E != nullptr && G != nullptr && M != nullptr)
    {
        calculateAdapter(G, E, M);
    }
}

void LatticeSquare::print()
{}