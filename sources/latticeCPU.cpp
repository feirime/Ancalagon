#include "lattice.h"

void LatticeCPU::latticeMalloc()
{
    x = new float[latticeSize];
    y = new float[latticeSize];
    mx = new float[latticeSize];
    my = new float[latticeSize];
    
    
    //!?
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
    int evenSize = 0;
    int oddSize = 0;
    if(layer % 2)
    {
        evenSize = mainLayerSize;
        oddSize = resultLayerSize;
    }
    else
    {
        evenSize = resultLayerSize;
        oddSize = mainLayerSize;
    }
    Geven = new long long int[evenSize];
    Eeven = new float[evenSize];
    Meven = new int[evenSize];
    Godd = new long long int[oddSize];
    Eodd = new float[oddSize];
    Modd = new int[oddSize];
    Gadd = new long long int[connectedSpinsSize];
    Eadd = new float[connectedSpinsSize];
    Madd = new int[connectedSpinsSize];
}

void LatticeCPU::addCalculate()
{
    
}

void LatticeCPU::calculate()
{
    for(auto i = 0; i < mainLayerSize; i++)
    {
        for(auto j = 0; j < connectedSpinsSize; j++)
        {
            float xij = mainLayerX[i] - connectedSpinsX[j];
            float yij = mainLayerY[i] - connectedSpinsY[j];
            float r = sqrt(xij * xij + yij * yij);
            Eeven[j + i * connectedSpinsSize] = (mainLayerMx[i] * connectedSpinsMx[j] + mainLayerMy[i] * connectedSpinsMy[j]) / (pow(r, 3)) 
                                      - 3 * (mainLayerMx[i] * xij + mainLayerMy[i] * yij) * (connectedSpinsMx[j] * xij + connectedSpinsMy[j] * yij) / (pow(r, 5));
            //M[j + i * connectedSpinsSize] = mainLayerM[i] + connectedSpinsM[j];
        }
    }
}

LatticeCPU::~LatticeCPU() 
{
    std::cout << "CPU destructor\n";
    delete [] Geven;
    delete [] Eeven;
    delete [] Meven;
    delete [] Godd;
    delete [] Eodd;
    delete [] Modd;
    delete [] Gadd;
    delete [] Eadd;
    delete [] Madd;
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
