#include "lattice.h"

void LatticeCPU::latticeMalloc()
{
    x = new float[latticeSize];
    y = new float[latticeSize];
    mx = new float[latticeSize];
    my = new float[latticeSize];
}

void LatticeCPU::createDOS()
{
    G = new long long int[latticeSize];
    E = new float[latticeSize];
    M = new int[latticeSize];
    //!?
}

void LatticeCPU::mainMapMaker()
{}

void LatticeCPU::calculate(float splitSeed)
{
    //int linearSize = (int)sqrt((float)latticeSize);
    //for(auto i = 0; i < mainLayerSize; i++)
    //{
    //    connectedMapMaker();
    //    for(auto j = 0; j < connectedSpinsSize; j++)
    //    {
    //        float xij = mainLayer.x[i] - connectedSpins.x[j];
    //        float yij = mainLayer.y[i] - connectedSpins.y[j];
    //        float r = sqrt(xij * xij + yij * yij);
    //        E[j + i * connectedSpins] = (mainLayer.mx[i] * connectedSpins.mx[j] + mainLayer.my[i] * connectedSpins.my[j]) / (pow(r, 3)) 
    //                                  - 3 * (mainLayer.mx[i] * xij + mainLayer.my[i] * yij) * (connectedSpins.mx[j] * xij + connectedSpins.my[j] * yij) / (pow(r, 5));
    //    }
    //}
    //double r, Xij, Yij;
    //int i = index / matrix_linear_size;
    //int j = index % matrix_linear_size;
    //if (index < matrix_linear_size * matrix_linear_size)
    //{
    //    if (i < j)
    //    {
    //        Xij = x[i] - x[j];
    //        Yij = y[i] - y[j];
    //        r = sqrt((double)(Xij * Xij + Yij * Yij));
    //        matrix[index] = ((mx[i] * mx[j] + my[i] * my[j]) / (r * r * r) - 3. * (mx[i] * Xij + my[i] * Yij) * (mx[j] * Xij + my[j] * Yij) / (r * r * r * r * r));
    //    }
    //}
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
