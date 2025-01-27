#include <latticeFactory.h>

//Lattice* LatticeFactory::createLattice(std::string latticeType) 
//{
//    if (latticeType == "square") 
//    {
//        return new LatticeSquare();
//    }
//    else 
//    {
//        return nullptr;
//    }
//}

void LatticeSquare::createLattice() 
{
    latticeConstructor(G, E, M);
};

LatticeSquare::~LatticeSquare() 
{
    latticeDestructor(G, E, M);
}
