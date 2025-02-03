
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

__global__ void test()
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
        printf("test GPU\n");
}

void mapMaker(double *x, double *y, double *mx, double *my, int latticeSize, float seed)
{
    double xMin = *std::min_element(x, x + latticeSize);
    double xMax = *std::max_element(x, x + latticeSize);
    double yMin = *std::min_element(y, y + latticeSize);
    double yMax = *std::max_element(y, y + latticeSize);

    for(auto i = 0; i < latticeSize; i++)
    {
        int j = 0;
        double xPrevious = x[j];
        while(x[j] == xPrevious)
        {
            j++;
        }
    }
}

__global__ void calculate(long long int *G, float *E, int *M, double *x, double *y, double *mx, double *my, int latticeSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
}
