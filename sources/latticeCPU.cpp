#include "lattice.h"

void LatticeCPU::latticeMalloc()
{
    x = new float[latticeSize];
    y = new float[latticeSize];
    mx = new float[latticeSize];
    my = new float[latticeSize];
}

void LatticeCPU::latticeMainMalloc()
{
    size_t evenSize = 0;
    if(layer % 2)
        evenSize = layerMainSize;
    else
        evenSize = layerAddSize;
    xMain = new float[evenSize];
    yMain = new float[evenSize];
    mxMain = new float[evenSize];
    myMain = new float[evenSize];
}

void LatticeCPU::latticeAddMalloc()
{
    size_t oddSize = 0;
    if(layer % 2)
        oddSize = layerAddSize;
    else
        oddSize = layerMainSize;
    xAdd = new float[oddSize];
    yAdd = new float[oddSize];
    mxAdd = new float[oddSize];
    myAdd = new float[oddSize];
}

void LatticeCPU::dosMalloc()
{
    delete[] Gmain;
    delete[] Emain;
    delete[] Mmain;
    Gmain = new long long int[layerMainSize];
    Emain = new float[layerMainSize];
    Mmain = new int[layerMainSize];
    Gmain = std::copy(Gmain, Gmain + layerMainSize, Gresult);
    Emain = std::copy(Emain, Emain + layerMainSize, Eresult);
    Mmain = std::copy(Mmain, Mmain + layerMainSize, Mresult);
    delete[] Gadd;
    delete[] Eadd;
    delete[] Madd;
    delete[] Gresult;
    delete[] Eresult;
    delete[] Mresult;
    Gadd = new long long int[layerAddSize];
    Eadd = new float[layerAddSize];
    Madd = new int[layerAddSize];
    Gresult = new long long int[layerResultSize];
    Eresult = new float[layerResultSize];
    Mresult = new int[layerResultSize];
}

void LatticeCPU::addCalculate()
{
    
}

void LatticeCPU::calculate()
{
    for(auto i = 0; i < layerMainSize; i++)
    {
        for(auto j = 0; j < layerAddSize; j++)
        {
            float xij = xMain[i] - xAdd[j];
            float yij = yMain[i] - yAdd[j];
            float r = sqrt(xij * xij + yij * yij);
            Eresult[j + i * layerAddSize] = (mxMain[i] * mxAdd[j] + myMain[i] * myAdd[j]) / (pow(r, 3)) 
                                      - 3 * (mxMain[i] * xij + myMain[i] * yij) * (mxAdd[j] * xij + myAdd[j] * yij) / (pow(r, 5));
            Mresult[j + i * layerAddSize] = Mmain[i] + Madd[j];
        }
    }
}

LatticeCPU::~LatticeCPU()
{
    std::cout << "CPU destructor\n";
    delete [] Gmain;
    delete [] Emain;
    delete [] Mmain;
    delete [] Gresult;
    delete [] Eresult;
    delete [] Mresult;
    delete [] Gadd;
    delete [] Eadd;
    delete [] Madd;
    delete [] x;
    delete [] y;
    delete [] mx;
    delete [] my;
    delete [] xMain;
    delete [] yMain;
    delete [] mxMain;
    delete [] myMain;
    delete [] xAdd;
    delete [] yAdd;
    delete [] mxAdd;
    delete [] myAdd;
}
