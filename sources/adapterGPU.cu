#include "adapterGPU.h"
#include "gpu.h"
#include <iostream>

void dosConstructorAdapter(unsigned long long *&G, float *&E, float *&M, 
    unsigned long long *&conf, int size)
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

void latticeConstructorAdapter(float *&x, float *&y, float *&mx, float *&my, int size)
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
