#include <latticeFactory.h>

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
    latticeDestructor(G, E, M);
}

void LatticeSquare::createLattice() 
{
    latticeConstructor(G, E, M);
};
