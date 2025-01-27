#include "adapterGPU.h"
#include "gpu.h"
#include <iostream>

void latticeConstructorAdapter(long long int *&G, float *&E, int *&M)
{
    printf("G = %p, E = %p, M = %p\n", G, E, M);
    cudaMallocManaged(&G, sizeof(G));
    cudaMallocManaged(&E, sizeof(E));
    cudaMallocManaged(&M, sizeof(M));
    printf("G = %p, E = %p, M = %p\n", G, E, M);
}

void latticeDestructorAdapter(long long int *&G, float *&E, int *&M)
{
    if(G != nullptr)
        cudaFree(G);
    if(E != nullptr)
        cudaFree(E);
    if(M != nullptr)
        cudaFree(M);
}

void calculateAdapter(long long int *&G, float *&E, int *&M)
{
    printf("test CPU\n");
    test<<<1, 1>>>();
    cudaDeviceSynchronize();
}
