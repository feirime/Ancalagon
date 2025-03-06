#include "adapterGPU.h"
#include "gpu.h"
#include <iostream>

void latticeConstructorDOSAdapter(long long int *&G, float *&E, int *&M)
{
    cudaMallocManaged(&G, sizeof(G));
    cudaMallocManaged(&E, sizeof(E));
    cudaMallocManaged(&M, sizeof(M));
}

void latticeConstructorAdapter(float *&x, float *&y, float *&mx, float *&my, int size)
{
    cudaMallocManaged(&x, size * sizeof(x));
    cudaMallocManaged(&y, size * sizeof(y));
    cudaMallocManaged(&mx, size * sizeof(mx));
    cudaMallocManaged(&my, size * sizeof(my));
}

void latticeDestructorAdapter(long long int *&G, float *&E, int *&M, float *&x, float *&y, float *&mx, float *&my)
{
    if(my != nullptr)
    {
        cudaFree(my);
        printf("free my\n");
    }
    if(mx != nullptr)
    {
        cudaFree(mx);
        printf("free mx\n");
    }
    if(y != nullptr)
    {
        cudaFree(y);
        printf("free y\n");
    }
    if(x != nullptr)
    {
        cudaFree(x);
        printf("free x\n");
    }
    if(M != nullptr)
    {
        cudaFree(M);
        printf("free M\n");
    }
    if(E != nullptr)
    {
        cudaFree(E);
        printf("free E\n");
    }
    if(G != nullptr)
    {
        cudaFree(G);
        printf("free G\n");
    }
}

void calculateAdapter(long long int *&G, float *&E, int *&M, float *&x, float *&y, float *&mx, float *&my, int latticeSize, float splitSeed)
{
    cudaDeviceProp dev{};
    cudaGetDeviceProperties(&dev, 0);
    static size_t block_dim = 512;
    static size_t grid_dim = get_SP_cores(dev);
    std::cout << "sp_cores: " << get_SP_cores(dev) << "\n";
    unsigned int *configuration;
    //mapMaker(x, y, mx, my, latticeSize, splitSeed);
    //unifing<<<grid_dim, block_dim>>>();
    cudaDeviceSynchronize();
}
