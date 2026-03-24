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

void LatticeCPU::dosMainMalloc()
{
    Gmain = new unsigned long long int[dosMainSize];
    Emain = new float[dosMainSize];
    Mmain = new float[dosMainSize];
    confMain = new long long int[dosMainSize];
}

void LatticeCPU::dosAddMalloc()
{
    Gadd = new unsigned long long int[dosAddSize];
    Eadd = new float[dosAddSize];
    Madd = new float[dosAddSize];
    confAdd = new long long int[dosAddSize];
}

void LatticeCPU::dosResultMalloc()
{
    Gresult = new unsigned long long int[dosResultSize];
    Eresult = new float[dosResultSize];
    Mresult = new float[dosResultSize];
    confResult = new long long int[dosResultSize];
}

void LatticeCPU::dosResultFree()
{
    delete[] Gresult;
    delete[] Eresult;
    delete[] Mresult;
    delete[] confResult;
}

void LatticeCPU::dosMainFree()
{
    delete[] Gmain;
    delete[] Emain;
    delete[] Mmain;
    delete[] confMain;
}

void LatticeCPU::dosAddFree()
{
    delete[] Gadd;
    delete[] Eadd;
    delete[] Madd;
    delete[] confAdd;
}

void LatticeCPU::calculateMain()
{
    for(auto confMain = 0; confMain < dosMainSize; confMain++)
    {
        Emain[confMain] = 0;
        Mmain[confMain] = 0;
        for(auto i = 0; i < layerMainSize; i++)
        {
            float mxi = mxMain[i] * (confMain >> i & 1 ? -1 : 1);
            float myi = myMain[i] * (confMain >> i & 1 ? -1 : 1);
            for(auto j = i + 1; j < layerMainSize; j++)
            {
                float distance = sqrt(pow(xMain[i] - xMain[j], 2) + pow(yMain[i] - yMain[j], 2));
                if(distance > iteractionRadius)
                    continue;
                float mxj = mxMain[j] * (confMain >> j & 1 ? -1 : 1);
                float myj = myMain[j] * (confMain >> j & 1 ? -1 : 1);
                float xij = xMain[i] - xMain[j];
                float yij = yMain[i] - yMain[j];
                float r = sqrt(xij * xij + yij * yij);
                Emain[confMain] += (mxi * mxj + myi * myj) / (pow(r, 3)) - 3 * (mxi * xij + myi * yij) * (mxj * xij + myj * yij) / (pow(r, 5));
            }
            Mmain[confMain] += (mxi + myi) * sqrt(2) / 2; //projection on 45 degree axis
        }
    }
}

void LatticeCPU::calculateAdd()
{
    for(auto confAdd = 0; confAdd < dosAddSize; confAdd++)
    {
        Eadd[confAdd] = 0;
        Madd[confAdd] = 0;
        for(auto i = 0; i < layerAddSize; i++)
        {
            float mxi = mxAdd[i] * (confAdd >> i & 1 ? -1 : 1);
            float myi = myAdd[i] * (confAdd >> i & 1 ? -1 : 1);
            for(auto j = i + 1; j < layerAddSize; j++)
            {
                float distance = sqrt(pow(xAdd[i] - xAdd[j], 2) + pow(yAdd[i] - yAdd[j], 2));
                if(distance > iteractionRadius)
                    continue;
                float mxj = mxAdd[j] * (confAdd >> j & 1 ? -1 : 1);
                float myj = myAdd[j] * (confAdd >> j & 1 ? -1 : 1);
                float xij = xAdd[i] - xAdd[j];
                float yij = yAdd[i] - yAdd[j];
                float r = sqrt(xij * xij + yij * yij);
                Eadd[confAdd] += (mxi * mxj + myi * myj) / (pow(r, 3)) - 3 * (mxi * xij + myi * yij) * (mxj * xij + myj * yij) / (pow(r, 5));
            }
            Madd[confAdd] += (mxi + myi) * sqrt(2) / 2; //projection on 45 degree axis
        }
    }
}

void LatticeCPU::calculateUnified()
{
    for(auto confMain = 0; confMain < dosMainSize; confMain++)
    {
        for(auto confAdd = 0; confAdd < dosAddSize; confAdd++)
        {
            Eresult[confMain + confAdd * dosMainSize] = 0;
            Mresult[confMain + confAdd * dosMainSize] = 0;
            for(auto i = 0; i < layerMainSize; i++)
            {
                for(auto j = i + 1; j < layerAddSize; j++)
                {
                    float xij = xMain[i] - xAdd[j];
                    float yij = yMain[i] - yAdd[j];
                    float r = sqrt(xij * xij + yij * yij);
                    Eresult[confMain + confAdd * dosMainSize] += (mxMain[i] * mxAdd[j] + myMain[i] * myAdd[j]) / (pow(r, 3)) 
                                              - 3 * (mxMain[i] * xij + myMain[i] * yij) 
                                              * (mxAdd[j] * xij + myAdd[j] * yij) / (pow(r, 5));
                    Mresult[confMain + confAdd * dosMainSize] += Mmain[i] + Madd[j];
                }
            }
        }
    }
}

void LatticeCPU::dosMallocBrutforce()
{
    dosResultSize = pow(2, latticeSize);
    layerResultSize = latticeSize;
    Gresult = new unsigned long long int[dosResultSize];
    Eresult = new float[dosResultSize];
    Mresult = new float[dosResultSize];
}

LatticeCPU::~LatticeCPU()
{
    //std::cout << "CPU destructor\n";
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
