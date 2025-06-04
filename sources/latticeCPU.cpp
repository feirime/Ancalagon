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
    xMain = new float[layerMainSize];
    yMain = new float[layerMainSize];
    mxMain = new float[layerMainSize];
    myMain = new float[layerMainSize];
}

void LatticeCPU::latticeAddMalloc()
{
    xAdd = new float[layerAddSize];
    yAdd = new float[layerAddSize];
    mxAdd = new float[layerAddSize];
    myAdd = new float[layerAddSize];
}

void LatticeCPU::dosCopyMalloc()
{
    if(!isStart())
    {
        delete[] Gmain;
        delete[] Emain;
        delete[] Mmain;
        delete[] confMain;
    }
    size_t dosMainSize = pow(2, layerMainSize);
    Gmain = new long long int[dosMainSize];
    Emain = new float[dosMainSize];
    Mmain = new int[dosMainSize];
    confMain = new long long int[dosMainSize];
    size_t dosResultSize = 0;
    if(!isStart())
    {
        dosResultSize = pow(2, layerResultSize);
        for(auto i = 0; i < dosMainSize; i++)
        {
            Gmain[i] = Gresult[i];
            Emain[i] = Eresult[i];
            Mmain[i] = Mresult[i];
        }
        delete[] Gadd;
        delete[] Eadd;
        delete[] Madd;
        delete[] confAdd;
        delete[] Gresult;
        delete[] Eresult;
        delete[] Mresult;
        delete[] confResult;
    }
    else
    {
        dosResultSize = pow(2, layerResultSize);
        for(auto i = 0; i < dosMainSize; i++)
        {
            Gmain[i] = 0;
            Emain[i] = 0;
            Mmain[i] = 0;
        }
    }
    size_t dosAddSize = pow(2, layerAddSize);
    Gadd = new long long int[dosAddSize];
    Eadd = new float[dosAddSize];
    Madd = new int[dosAddSize];
    confAdd = new long long int[dosAddSize];
    Gresult = new long long int[dosResultSize];
    Eresult = new float[dosResultSize];
    Mresult = new int[dosResultSize];
    confResult = new long long int[dosResultSize];
}

void LatticeCPU::calculateMain()
{}

void LatticeCPU::calculateAdd()
{}

void LatticeCPU::calculateUnified()
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
