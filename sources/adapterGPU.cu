#include "adapterGPU.h"
#include "gpu.h"
#include <iostream>

void latticeConstructorDOSAdapter(long long int *&G, float *&E, int *&M, int size)
{
    cudaMallocManaged(&G, size * sizeof(G));
    cudaMallocManaged(&E, size * sizeof(E));
    cudaMallocManaged(&M, size * sizeof(M));
}

void latticeConstructorAdapter(float *&x, float *&y, float *&mx, float *&my, int size)
{
    cudaMallocManaged(&x, size * sizeof(x));
    cudaMallocManaged(&y, size * sizeof(y));
    cudaMallocManaged(&mx, size * sizeof(mx));
    cudaMallocManaged(&my, size * sizeof(my));
}

void latticeDestructorAdapter(long long int *&Gmain, float *&Emain, int *&Mmain, 
    long long int *&Gresult, float *&Eresult, int *&Mresult,  float *&x, float *&y, float *&mx, float *&my)
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
    if(Mmain != nullptr)
    {
        cudaFree(Mmain);
        printf("free Mmain\n");
    }
    if(Emain != nullptr)
    {
        cudaFree(Emain);
        printf("free Emain\n");
    }
    if(Gmain != nullptr)
    {
        cudaFree(Gmain);
        printf("free Gmain\n");
    }
    if(Mresult != nullptr)
    {
        cudaFree(Mresult);
        printf("free Mresult\n");
    }
    if(Eresult != nullptr)
    {
        cudaFree(Eresult);
        printf("free Eresult\n");
    }
    if(Gresult != nullptr)
    {
        cudaFree(Gresult);
        printf("free Gresult\n");
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
