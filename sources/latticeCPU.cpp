#include "lattice.h"

void LatticeCPU::latticeMalloc()
{
    x = new float[latticeSize];
    y = new float[latticeSize];
    mx = new float[latticeSize];
    my = new float[latticeSize];
    mainLayerX = new float[latticeSize];
    mainLayerY = new float[latticeSize];
    mainLayerMx = new float[latticeSize];
    mainLayerMy = new float[latticeSize];
    connectedSpinsX = new float[latticeSize];
    connectedSpinsY = new float[latticeSize];
    connectedSpinsMx = new float[latticeSize];
    connectedSpinsMy = new float[latticeSize];
}

void LatticeCPU::dosMalloc()
{
    G = new long long int[latticeSize];
    E = new float[latticeSize];
    M = new int[latticeSize];
    //!?
}

unsigned int LatticeCPU::mainMapMaker()
{
    return 0;
}

unsigned int LatticeCPU::connectedMapMaker()
{
    return 0;
}

void LatticeCPU::calculate()
{
    int linearSize = (int)sqrt((float)latticeSize);
    auto mainLayerSize = mainMapMaker();
    for(auto i = 0; i < mainLayerSize; i++)
    {
        auto connectedSpinsSize = connectedMapMaker();
        for(auto j = 0; j < connectedSpinsSize; j++)
        {
            float xij = mainLayerX[i] - connectedSpinsX[j];
            float yij = mainLayerY[i] - connectedSpinsY[j];
            float r = sqrt(xij * xij + yij * yij);
            E[j + i * connectedSpinsSize] = (mainLayerMx[i] * connectedSpinsMx[j] + mainLayerMy[i] * connectedSpinsMy[j]) / (pow(r, 3)) 
                                      - 3 * (mainLayerMx[i] * xij + mainLayerMy[i] * yij) * (connectedSpinsMx[j] * xij + connectedSpinsMy[j] * yij) / (pow(r, 5));
            //M[j + i * connectedSpinsSize] = mainLayerM[i] + connectedSpinsM[j];
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
    delete [] mainLayerX;
    delete [] mainLayerY;
    delete [] mainLayerMx;
    delete [] mainLayerMy;
    delete [] connectedSpinsX;
    delete [] connectedSpinsY;
    delete [] connectedSpinsMx;
    delete [] connectedSpinsMy;
}
