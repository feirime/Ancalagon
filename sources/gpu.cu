
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

//__global__ void unifingSquare(int rightLayer, Lattice *left, int leftSize, Lattice *right, int rightSize, Lattice *result, float *JMap) 
//{
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    for(int i = 0; i < leftSize; i++)
//    {
//        for(int j = 0; j < rightSize; j++)
//        {
//            result[j].M = left[i].M + right[j].M;
//            for(int bound = 0; bound < 1 + rightLayer * 3; bound++)
//            {
//                result[j].E = left[i].E + right[j].E + JMap[i];
//            }
//            result[j].G = left[i].G;
//        }
//    }
//}

__global__ void test()
{
    printf("test GPU\n");
}
