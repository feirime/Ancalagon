#include "adapterGPU.h"
#include "gpu.h"
#include <iostream>

void latticeConstructorAdapter(long long int *&G, float *&E, int *&M)
{
    cudaMallocManaged(&G, sizeof(G));
    cudaMallocManaged(&E, sizeof(E));
    cudaMallocManaged(&M, sizeof(M));
}

void latticeDestructorAdapter(long long int *&G, float *&E, int *&M)
{
    if(G != nullptr)
        cudaFree(G);
        printf("free G\n");
    if(E != nullptr)
        cudaFree(E);
        printf("free E\n");
    if(M != nullptr)
        cudaFree(M);
        printf("free M\n");
}

void calculateAdapter(long long int *&G, float *&E, int *&M)
{
    cudaDeviceProp dev{};
    cudaGetDeviceProperties(&dev, 0);
    static size_t block_dim = 512;
    static size_t grid_dim = get_SP_cores(dev);
    std::cout << "sp_cores: " << get_SP_cores(dev) << "\n";
    cudaDeviceSynchronize();
}
