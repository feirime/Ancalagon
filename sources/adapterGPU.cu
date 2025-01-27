#include "adapterGPU.h"
#include "gpu.h"
#include <iostream>

void testAdapterGPU()
{
    printf("test CPU\n");
    test<<<1, 1>>>();
    cudaDeviceSynchronize();
}

void latticeConstructor(long long int *G, float *E, int *M)
{
    cudaMallocManaged(&G, sizeof(G));
    cudaMallocManaged(&E, sizeof(E));
    cudaMallocManaged(&M, sizeof(M));
}

void latticeDestructor(long long int *G, float *E, int *M)
{
    cudaFree(G);
    cudaFree(E);
    cudaFree(M);
}
