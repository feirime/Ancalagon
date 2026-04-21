#include "lattice.h"

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
    Gmain = new unsigned long long[dosMainSize];
    Emain = new float[dosMainSize];
    Mmain = new float[dosMainSize];
    confMain = new unsigned long long[dosMainSize];
}

void LatticeCPU::dosAddMalloc()
{
    Gadd = new unsigned long long[dosAddSize];
    Eadd = new float[dosAddSize];
    Madd = new float[dosAddSize];
    confAdd = new unsigned long long[dosAddSize];
}

void LatticeCPU::dosResultMalloc()
{
    Gresult = new unsigned long long[dosResultSize];
    Eresult = new float[dosResultSize];
    Mresult = new float[dosResultSize];
    confResult = new unsigned long long[dosResultSize];
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
    for(auto confMainIdx = 0; confMainIdx < dosMainSize; confMainIdx++)
    {
        Emain[confMainIdx] = 0;
        Mmain[confMainIdx] = 0;
        for(auto i = 0; i < layerMainSize; i++)
        {
            float mxi = mxMain[i] * (confMainIdx >> i & 1 ? -1 : 1);
            float myi = myMain[i] * (confMainIdx >> i & 1 ? -1 : 1);
            for(auto j = i + 1; j < layerMainSize; j++)
            {
                float distance = sqrt(pow(xMain[i] - xMain[j], 2) + pow(yMain[i] - yMain[j], 2));
                if(distance > iteractionRadius)
                    continue;
                float mxj = mxMain[j] * (confMainIdx >> j & 1 ? -1 : 1);
                float myj = myMain[j] * (confMainIdx >> j & 1 ? -1 : 1);
                float xij = xMain[i] - xMain[j];
                float yij = yMain[i] - yMain[j];
                float r = sqrt(xij * xij + yij * yij);
                Emain[confMainIdx] += (mxi * mxj + myi * myj) / (pow(r, 3)) - 3 * (mxi * xij + myi * yij) * (mxj * xij + myj * yij) / (pow(r, 5));
            }
            Mmain[confMainIdx] += (mxi + myi) * sqrt(2) / 2; //projection on 45 degree axis
        }
        confMain[confMainIdx] = confMainIdx;
    }
}

void LatticeCPU::calculateAdd()
{
    for(auto confAddidx = 0; confAddidx < dosAddSize; confAddidx++)
    {
        Eadd[confAddidx] = 0;
        Madd[confAddidx] = 0;
        for(auto i = 0; i < layerAddSize; i++)
        {
            float mxi = mxAdd[i] * (confAddidx >> i & 1 ? -1 : 1);
            float myi = myAdd[i] * (confAddidx >> i & 1 ? -1 : 1);
            for(auto j = i + 1; j < layerAddSize; j++)
            {
                float distance = sqrt(pow(xAdd[i] - xAdd[j], 2) + pow(yAdd[i] - yAdd[j], 2));
                if(distance > iteractionRadius)
                    continue;
                float mxj = mxAdd[j] * (confAddidx >> j & 1 ? -1 : 1);
                float myj = myAdd[j] * (confAddidx >> j & 1 ? -1 : 1);
                float xij = xAdd[i] - xAdd[j];
                float yij = yAdd[i] - yAdd[j];
                float r = sqrt(xij * xij + yij * yij);
                Eadd[confAddidx] += (mxi * mxj + myi * myj) / (pow(r, 3)) - 3 * (mxi * xij + myi * yij) * (mxj * xij + myj * yij) / (pow(r, 5));
            }
            Madd[confAddidx] += (mxi + myi) * sqrt(2) / 2; //projection on 45 degree axis
        }
        confAdd[confAddidx] = confAddidx;
    }
}

void LatticeCPU::calculateUnified()
{
    for(auto confMainIdx = 0; confMainIdx < dosMainSize; confMainIdx++)
    {
        for(auto confAddIdx = 0; confAddIdx < dosAddSize; confAddIdx++)
        {
            Eresult[confMainIdx + confAddIdx * dosMainSize] = 0;
            Mresult[confMainIdx + confAddIdx * dosMainSize] = 0;
            for(auto i = 0; i < layerMainSize; i++)
            {
                for(auto j = i + 1; j < layerAddSize; j++)
                {
                    float distance = sqrt(pow(xAdd[i] - xAdd[j], 2) + pow(yAdd[i] - yAdd[j], 2));
                    if(distance > iteractionRadius)
                        continue;
                    float xij = xMain[i] - xAdd[j];
                    float yij = yMain[i] - yAdd[j];
                    float r = sqrt(xij * xij + yij * yij);
                    Eresult[confMainIdx + confAddIdx * dosMainSize] += (mxMain[i] * mxAdd[j] + myMain[i] * myAdd[j]) / (pow(r, 3)) 
                                              - 3 * (mxMain[i] * xij + myMain[i] * yij) 
                                              * (mxAdd[j] * xij + myAdd[j] * yij) / (pow(r, 5));
                    Mresult[confMainIdx + confAddIdx * dosMainSize] += Mmain[i] + Madd[j];
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
    delete [] xMain;
    delete [] yMain;
    delete [] mxMain;
    delete [] myMain;
    delete [] xAdd;
    delete [] yAdd;
    delete [] mxAdd;
    delete [] myAdd;
}
