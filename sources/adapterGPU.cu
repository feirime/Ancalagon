#include "adapterGPU.h"
#include "gpu.h"
#include <iostream>

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
    cudaMallocManaged(&G, size * sizeof(G));
    cudaMallocManaged(&E, size * sizeof(E));
    cudaMallocManaged(&M, size * sizeof(M));
    cudaMallocManaged(&conf, size * sizeof(conf));
}

void dosDestructorAdapter(unsigned long long *&G, float *&E, float *&M, 
    unsigned long long *&conf)
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
    cudaMallocManaged(&x, size * sizeof(x));
    cudaMallocManaged(&y, size * sizeof(y));
    cudaMallocManaged(&mx, size * sizeof(mx));
    cudaMallocManaged(&my, size * sizeof(my));
}

void latticeDestructorAdapter(float *&x, float *&y, float *&mx, float *&my)
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
}

void kernelElementaryAdapter(float *&x, float *&y, float *&mx, float *&my, int latticeSize,
    unsigned long long *&G, float *&E, float *&M, unsigned long long *&conf, size_t dosSize, float iteractionRadius)
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    ElementaryClalc<<<get_SP_cores(devProp), 16>>>(x, y, mx, my, latticeSize, G, E, M, conf, dosSize, iteractionRadius);
    cudaDeviceSynchronize();
}


void kernelUnifyingAdapter(float *&xMain, float *&yMain, float *&mxMain, float *&myMain, size_t latticeMainSize,
    float *&xAdd, float *&yAdd, float *&mxAdd, float *&myAdd, size_t latticeAddSize,
    unsigned long long *&Gmain, float *&Emain, float *&Mmain, unsigned long long *&confMain, size_t dosMainSize,
    unsigned long long *&Gadd, float *&Eadd, float *&Madd, unsigned long long *&confAdd, size_t dosAddSize,
    unsigned long long *&Gresult, float *&Eresult, float *&Mresult, unsigned long long *&confResult, size_t dosResultSize, 
    float iteractionRadius)
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    unifing<<<get_SP_cores(devProp), 16>>>(xMain, yMain, mxMain, myMain, latticeMainSize,
        xAdd, yAdd, mxAdd, myAdd, latticeAddSize,
        Gmain, Emain, Mmain, confMain, dosMainSize,
        Gadd, Eadd, Madd, confAdd, dosAddSize,
        Gresult, Eresult, Mresult, confResult, dosResultSize, iteractionRadius);
    cudaDeviceSynchronize();
}
