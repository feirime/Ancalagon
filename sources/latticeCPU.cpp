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

void LatticeCPU::dosMalloc()
{
    delete[] Gmain;
    delete[] Emain;
    delete[] Mmain;
    size_t dosMainSize = pow(2, layerMainSize);
    Gmain = new long long int[dosMainSize];
    Emain = new float[dosMainSize];
    Mmain = new int[dosMainSize];
    Gmain = std::copy(Gmain, Gmain + dosMainSize, Gresult);
    Emain = std::copy(Emain, Emain + dosMainSize, Eresult);
    Mmain = std::copy(Mmain, Mmain + dosMainSize, Mresult);
    delete[] Gadd;
    delete[] Eadd;
    delete[] Madd;
    delete[] Gresult;
    delete[] Eresult;
    delete[] Mresult;
    size_t dosAddSize = pow(2, layerAddSize);
    Gadd = new long long int[dosAddSize];
    Eadd = new float[dosAddSize];
    Madd = new int[dosAddSize];
    size_t layerResultSize = pow(2, layerResultSize);
    Gresult = new long long int[layerResultSize];
    Eresult = new float[layerResultSize];
    Mresult = new int[layerResultSize];
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
