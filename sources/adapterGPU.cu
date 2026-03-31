#include "adapterGPU.h"
#include "gpu.h"
#include <iostream>

namespace
{
    void checkCuda(cudaError_t status, const char *context)
    {
        if(status != cudaSuccess)
        {
            std::cerr << context << ": " << cudaGetErrorString(status) << '\n';
            std::exit(1);
        }
    }
}

void dosConstructorAdapter(unsigned long long *&G, float *&E, float *&M, 
    unsigned long long *&conf, size_t size)
{
    if(G != nullptr)
    {
        cudaFree(G);
        //printf("free G\n");
    }
    if(E != nullptr)
    {
        cudaFree(E);
        //printf("free E\n");
    }
    if(M != nullptr)
    {
        cudaFree(M);
        //printf("free M\n");
    }
    if(conf != nullptr)
    {
        cudaFree(conf);
        //printf("free conf\n");
    }
    cudaMallocManaged(&G, size * sizeof(*G));
    cudaMallocManaged(&E, size * sizeof(*E));
    printf("e has %lu bytes\n", size * sizeof(*E));
    printf("e in address %p\n", E);
    cudaMallocManaged(&M, size * sizeof(*M));
    cudaMallocManaged(&conf, size * sizeof(*conf));
}

void dosDestructorAdapter(unsigned long long *&G, float *&E, float *&M, 
    unsigned long long *&conf)
{
    if(G != nullptr)
    {
        cudaFree(G);
        G = nullptr;
        //printf("free G\n");
    }
    if(E != nullptr)
    {
        cudaFree(E);
        E = nullptr;
        //printf("free E\n");
    }
    if(M != nullptr)
    {
        cudaFree(M);
        M = nullptr;
        //printf("free M\n");
    }
    if(conf != nullptr)
    {
        cudaFree(conf);
        conf = nullptr;
        //printf("free conf\n");
    }
}

void latticeConstructorAdapter(float *&x, float *&y, float *&mx, float *&my, size_t size)
{
    if(x != nullptr)
    {
        cudaFree(x);
        //printf("free x\n");
    }
    if(y != nullptr)
    {
        cudaFree(y);
        //printf("free y\n");
    }
    if(mx != nullptr)
    {
        cudaFree(mx);
        //printf("free mx\n");
    }
    if(my != nullptr)
    {
        cudaFree(my);
        //printf("free my\n");
    }
    cudaMallocManaged(&x, size * sizeof(*x));
    cudaMallocManaged(&y, size * sizeof(*y));
    cudaMallocManaged(&mx, size * sizeof(*mx));
    cudaMallocManaged(&my, size * sizeof(*my));
}

void latticeDestructorAdapter(float *&x, float *&y, float *&mx, float *&my)
{
    if(x != nullptr)
    {
        cudaFree(x);
        x = nullptr;
        //printf("free x\n");
    }
    if(y != nullptr)
    {
        cudaFree(y);
        y = nullptr;
        //printf("free y\n");
    }
    if(mx != nullptr)
    {
        cudaFree(mx);
        mx = nullptr;
        //printf("free mx\n");
    }
    if(my != nullptr)
    {
        cudaFree(my);
        my = nullptr;
        //printf("free my\n");
    }
}

void kernelElementaryAdapter(float *x, float *y, float *mx, float *my, int layerSize,
    unsigned long long *G, float *E, float *M, unsigned long long *conf, size_t dosSize, size_t confSize, float iteractionRadius)
{
    cudaDeviceProp devProp;
    checkCuda(cudaGetDeviceProperties(&devProp, 0), "cudaGetDeviceProperties failed in kernelElementaryAdapter");
    static size_t block_dim = 128;
    static size_t grid_dim = get_SP_cores(devProp) / block_dim;
    ElementaryClalc<<<grid_dim, block_dim>>>(x, y, mx, my, layerSize, G, E, M, conf, dosSize, confSize, iteractionRadius);
    checkCuda(cudaGetLastError(), "ElementaryClalc launch failed");
    checkCuda(cudaDeviceSynchronize(), "ElementaryClalc execution failed");
}


void kernelUnifyingAdapter(float *xMain, float *yMain, float *mxMain, float *myMain, size_t latticeMainSize,
    float *xAdd, float *yAdd, float *mxAdd, float *myAdd, size_t latticeAddSize,
    unsigned long long *Gmain, float *Emain, float *Mmain, unsigned long long *confMain, size_t dosMainSize, size_t confMainSize,
    unsigned long long *Gadd, float *Eadd, float *Madd, unsigned long long *confAdd, size_t dosAddSize, size_t confAddSize,
    unsigned long long *Gresult, float *Eresult, float *Mresult, unsigned long long *confResult, size_t dosResultSize, size_t confResultSize,
    float iteractionRadius)
{
    cudaDeviceProp devProp;
    checkCuda(cudaGetDeviceProperties(&devProp, 0), "cudaGetDeviceProperties failed in kernelUnifyingAdapter");
    static size_t block_dim = 128;
    static size_t grid_dim = get_SP_cores(devProp) / block_dim;
    unifing<<<grid_dim, block_dim>>>(xMain, yMain, mxMain, myMain, latticeMainSize,
        xAdd, yAdd, mxAdd, myAdd, latticeAddSize,
        Gmain, Emain, Mmain, confMain, dosMainSize, confMainSize,
        Gadd, Eadd, Madd, confAdd, dosAddSize, confAddSize,
        Gresult, Eresult, Mresult, confResult, dosResultSize, confResultSize,
        iteractionRadius);
    checkCuda(cudaGetLastError(), "unifing launch failed");
    checkCuda(cudaDeviceSynchronize(), "unifing execution failed");
}
