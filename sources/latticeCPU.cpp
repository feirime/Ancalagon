#include "lattice.h"

void LatticeCPU::latticeMalloc()
{
    x = new float[latticeSize];
    y = new float[latticeSize];
    mx = new float[latticeSize];
    my = new float[latticeSize];
}

void LatticeCPU::latticeEvenMalloc()
{
    size_t evenSize = 0;
    if(layer % 2)
        evenSize = layerMainSize;
    else
        evenSize = layerConnectedSize;
    xEven = new float[evenSize];
    yEven = new float[evenSize];
    mxEven = new float[evenSize];
    myEven = new float[evenSize];
}

void LatticeCPU::latticeOddMalloc()
{
    size_t oddSize = 0;
    if(layer % 2)
        oddSize = layerConnectedSize;
    else
        oddSize = layerMainSize;
    xOdd = new float[oddSize];
    yOdd = new float[oddSize];
    mxOdd = new float[oddSize];
    myOdd = new float[oddSize];
}

void LatticeCPU::dosMalloc()
{
    size_t evenSize = 0;
    size_t oddSize = 0;
    if(layer % 2)
    {
        evenSize = layerMainSize;
        oddSize = layerResultSize;
    }
    else
    {
        evenSize = layerResultSize;
        oddSize = layerMainSize;
    }
    Geven = new long long int[evenSize];
    Eeven = new float[evenSize];
    Meven = new int[evenSize];
    Godd = new long long int[oddSize];
    Eodd = new float[oddSize];
    Modd = new int[oddSize];
    Gadd = new long long int[layerConnectedSize];
    Eadd = new float[layerConnectedSize];
    Madd = new int[layerConnectedSize];
}

void LatticeCPU::addCalculate()
{
    
}

void LatticeCPU::calculate()
{
    for(auto i = 0; i < layerMainSize; i++)
    {
        for(auto j = 0; j < layerConnectedSize; j++)
        {
            float xij = xEven[i] - xOdd[j];
            float yij = yEven[i] - yOdd[j];
            float r = sqrt(xij * xij + yij * yij);
            Eeven[j + i * layerConnectedSize] = (mxEven[i] * mxOdd[j] + myEven[i] * myOdd[j]) / (pow(r, 3)) 
                                      - 3 * (mxEven[i] * xij + myEven[i] * yij) * (mxOdd[j] * xij + myOdd[j] * yij) / (pow(r, 5));
            //M[j + i * layerConnectedSize] = mainLayerM[i] + connectedSpinsM[j];
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
    delete [] xEven;
    delete [] yEven;
    delete [] mxEven;
    delete [] myEven;
    delete [] xOdd;
    delete [] yOdd;
    delete [] mxOdd;
    delete [] myOdd;
}
