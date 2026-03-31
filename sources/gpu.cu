
#include <iostream>
#include "adapterGPU.h"
#include "gpu.h"

int get_SP_cores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

__global__ void ElementaryClalc(float *x, float *y, float *mx, float *my, int latticeSize,
    unsigned long long *G, float *E, float *M, unsigned long long *conf, size_t dosSize, size_t confSize, float iteractionRadius)
{
    size_t globThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for(size_t confIdx = globThreadIdx; confIdx < confSize; confIdx += blockDim.x * gridDim.x)
    {
        float mxi;
        float myi;
        float mxj;
        float myj;
        G[confIdx] = 0;
        E[confIdx] = 0;
        M[confIdx] = 0;
        for(size_t i = 0; i < latticeSize; i++)
        {
            int confBitTemp = confIdx;
            mxi = mx[i] * (confBitTemp >> i & 1 ? -1 : 1);
            myi = my[i] * (confBitTemp >> i & 1 ? -1 : 1);
            for(size_t j = i + 1; j < latticeSize; j++)
            {
                float distance = sqrt(pow(x[i] - x[j], 2) + pow(y[i] - y[j], 2));
                if(distance > iteractionRadius)
                    continue;
                mxj = mx[j] * (confBitTemp >> j & 1 ? -1 : 1);
                myj = my[j] * (confBitTemp >> j & 1 ? -1 : 1);
                float xij = x[i] - x[j];
                float yij = y[i] - y[j];
                float r = sqrt(xij * xij + yij * yij);
                atomicAdd(&E[confIdx], (mxi * mxj + myi * myj) / (r * r * r) 
                                              - 3 * (mxi * xij + myi * yij) * (mxj * xij + myj * yij) / (r * r * r * r * r));
                G[confIdx] = 1;
            }
            atomicAdd(&M[confIdx], (mxi + myi) * sqrtf(2) / 2); //projection on 45 degree axis
        }
        conf[confIdx] = confIdx;
    }
}

__global__ void unifing(float *xMain, float *yMain, float *mxMain, float *myMain, size_t layerMainSize,
    float *xAdd, float *yAdd, float *mxAdd, float *myAdd, size_t layerAddSize,
    unsigned long long *Gmain, float *Emain, float *Mmain, unsigned long long *confMain, size_t dosMainSize, size_t confMainSize,
    unsigned long long *Gadd, float *Eadd, float *Madd, unsigned long long *confAdd, size_t dosAddSize, size_t confAddSize,
    unsigned long long *Gresult, float *Eresult, float *Mresult, unsigned long long *confResult, size_t dosResultSize, size_t confResultSize,
    float iteractionRadius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(size_t confMainIdx = blockIdx.x; confMainIdx < confMainSize; confMainIdx += gridDim.x)
    {
        for(size_t confAddIdx = threadIdx.x; confAddIdx < confAddSize; confAddIdx += blockDim.x)
        {
            Gresult[confMainIdx + confAddIdx * confMainSize] = 0;
            Eresult[confMainIdx + confAddIdx * confMainSize] = 0;
            Mresult[confMainIdx + confAddIdx * confMainSize] = 0;
            for(size_t i = 0; i < layerMainSize; i++)
            {
                for(size_t j = 0; j < layerAddSize; j++)
                {
                    float distance = sqrt((xMain[i] - xAdd[j]) * (xMain[i] - xAdd[j]) + (yMain[i] - yAdd[j]) * (yMain[i] - yAdd[j]));
                    if(distance > iteractionRadius)
                        continue;
                    float xij = xMain[i] - xAdd[j];
                    float yij = yMain[i] - yAdd[j];
                    float r = sqrt(xij * xij + yij * yij);
                    atomicAdd(&Eresult[confMainIdx + confAddIdx * confMainSize], (mxMain[i] * mxAdd[j] + myMain[i] * myAdd[j]) / (r * r * r) 
                                              - 3 * (mxMain[i] * xij + myMain[i] * yij) * (mxAdd[j] * xij + myAdd[j] * yij) / (r * r * r * r * r));
                    Gresult[confMainIdx + confAddIdx * confMainSize] = 1;
                }

            }
            atomicAdd(&Mresult[confMainIdx + confAddIdx * confMainSize], Mmain[confMainIdx] + Madd[confAddIdx]);
            printf("G[%llu] = %llu E[%llu] = %f\n", confMainIdx + confAddIdx * confMainSize, Gresult[confMainIdx + confAddIdx * confMainSize], 
                confMainIdx + confAddIdx * confMainSize, Eresult[confMainIdx + confAddIdx * confMainSize]);
            confResult[confMainIdx + confAddIdx * confMainSize] = confAdd[confAddIdx];
        }
    }
}
