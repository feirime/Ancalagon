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
    Mmain = new float[dosMainSize];
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
    Madd = new float[dosAddSize];
    confAdd = new long long int[dosAddSize];
    std::cout << "dosResultSize: " << dosResultSize << std::endl;
    Gresult = new long long int[dosResultSize];
    Eresult = new float[dosResultSize];
    Mresult = new float[dosResultSize];
    confResult = new long long int[dosResultSize];
}

void LatticeCPU::calculateMain()
{
    for(auto conf = 0; conf < pow(2, layerMainSize); conf++)
    {
        for(auto i = 0; i < layerMainSize; i++)
        {
            for(auto j = 0; j < layerMainSize; j++)
            {
                if(i == j)
                    continue;
                float distance = sqrt(pow(xMain[i] - xMain[j], 2) 
                                + pow(yMain[i] - yMain[j], 2));
                if(distance > iteractionRadius)
                    continue;
                float xij = xMain[i] - xMain[j];
                float yij = yMain[i] - yMain[j];
                float r = sqrt(xij * xij + yij * yij);
                Emain[i + j * layerMainSize] = (mxMain[i] * mxMain[j] + myMain[i] * myMain[j]) / (pow(r, 3)) 
                                          - 3 * (mxMain[i] * xij + myMain[i] * yij) 
                                          * (mxMain[j] * xij + myMain[j] * yij) / (pow(r, 5));
                Mmain[i + j * layerMainSize] = Mmain[i] + Mmain[j];
            }
        }
        //Здесь нужен переворот спина
    }
}

void LatticeCPU::calculateAdd()
{
    for(auto i = 0; i < layerAddSize; i++)
    {
        for(auto j = 0; j < layerAddSize; j++)
        {
            if(i == j)
                continue;
            float xij = xAdd[i] - xAdd[j];
            float yij = yAdd[i] - yAdd[j];
            float r = sqrt(xij * xij + yij * yij);
            Eadd[i + j * layerAddSize] = (mxAdd[i] * mxAdd[j] + myAdd[i] * myAdd[j]) / (pow(r, 3)) 
                                      - 3 * (mxAdd[i] * xij + myAdd[i] * yij) 
                                      * (mxAdd[j] * xij + myAdd[j] * yij) / (pow(r, 5));
            Madd[i + j * layerAddSize] = Madd[i] + Madd[j];
        }
    }
}

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
                                      - 3 * (mxMain[i] * xij + myMain[i] * yij) 
                                      * (mxAdd[j] * xij + myAdd[j] * yij) / (pow(r, 5));
            Mresult[j + i * layerAddSize] = Mmain[i] + Madd[j];
        }
    }
}

void LatticeCPU::dosMallocBrutforce()
{
    confs = pow(2, latticeSize);
    layerResultSize = latticeSize;
    Gresult = new long long int[confs];
    Eresult = new float[confs];
    Mresult = new float[confs];
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
