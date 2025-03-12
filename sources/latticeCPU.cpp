#include "lattice.h"

void LatticeCPU::latticeMalloc()
{
    x = new float[latticeSize];
    y = new float[latticeSize];
    mx = new float[latticeSize];
    my = new float[latticeSize];
}

void LatticeCPU::createDOS(int latticeSize)
{
    G = new long long int[latticeSize];
    E = new float[latticeSize];
    M = new int[latticeSize];
    //!?
}

void LatticeCPU::calculate(int latticeSize, float splitSeed)
{
    //E[] += ((mx[] * my[]) / () - ) //!?
}

LatticeCPU::~LatticeCPU() 
{
    std::cout << "CPU destructor\n";
    delete [] G;
    delete [] E;
    delete [] M;
    delete [] x;
    delete [] y;
    delete [] mx;
    delete [] my;
}
