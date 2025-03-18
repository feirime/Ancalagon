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
    int lienarSize = (int)sqrt((float)latticeSize)
    for(auto i = 0; i < linearSize; i++)
    {
        
    }

    double r, Xij, Yij;
    int i = index / matrix_linear_size;
    int j = index % matrix_linear_size;
    if (index < matrix_linear_size * matrix_linear_size)
    {
        if (i < j)
        {
            Xij = x[i] - x[j];
            Yij = y[i] - y[j];
            r = sqrt((double)(Xij * Xij + Yij * Yij));

            matrix[index] = ((mx[i] * mx[j] + my[i] * my[j]) / (r * r * r) - 3. * (mx[i] * Xij + my[i] * Yij) * (mx[j] * Xij + my[j] * Yij) / (r * r * r * r * r));
        }
    }
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
