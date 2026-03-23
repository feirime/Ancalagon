#include "adapterGPU.h"
#include "gpu.h"
#include <iostream>

void latticeConstructorDOSAdapter(unsigned long long int *&G, float *&E, float *&M, int size)
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
    cudaMallocManaged(&G, size * sizeof(G));
    cudaMallocManaged(&E, size * sizeof(E));
    cudaMallocManaged(&M, size * sizeof(M));
}

void latticeDestructorDOSAdapter(unsigned long long int *&G, float *&E, float *&M)
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
}

void latticeConstructorAdapter(float *&x, float *&y, float *&mx, float *&my, int size)
{
    cudaMallocManaged(&x, size * sizeof(x));
    cudaMallocManaged(&y, size * sizeof(y));
    cudaMallocManaged(&mx, size * sizeof(mx));
    cudaMallocManaged(&my, size * sizeof(my));
}

void latticeMainConstructorAdapter(float *&xMain, float *&yMain, float *&mxMain, float *&myMain, int sizeMain)
{
    if(myMain != nullptr)
    {
        cudaFree(myMain);
        //printf("free myMain\n");
    }
    if(mxMain != nullptr)
    {
        cudaFree(mxMain);
        //printf("free mxMain\n");
    }
    if(yMain != nullptr)
    {
        cudaFree(yMain);
        //printf("free yMain\n");
    }
    if(xMain != nullptr)
    {
        cudaFree(xMain);
        //printf("free xMain\n");
    }
    cudaMallocManaged(&xMain, sizeMain * sizeof(float));
    cudaMallocManaged(&yMain, sizeMain * sizeof(float));
    cudaMallocManaged(&mxMain, sizeMain * sizeof(float));
    cudaMallocManaged(&myMain, sizeMain * sizeof(float));
}

void latticeAddConstructorAdapter(float *&xAdd, float *&yAdd, float *&mxAdd, float *&myAdd, int sizeAdd)
{
    if(myAdd != nullptr)
    {
        cudaFree(myAdd);
        //printf("free myAdd\n");
    }
    if(mxAdd != nullptr)
    {
        cudaFree(mxAdd);
        //printf("free mxAdd\n");
    }
    if(yAdd != nullptr)
    {
        cudaFree(yAdd);
        //printf("free yAdd\n");
    }
    if(xAdd != nullptr)
    {
        cudaFree(xAdd);
        //printf("free xAdd\n");
    }
    cudaMallocManaged(&xAdd, sizeAdd * sizeof(float));
    cudaMallocManaged(&yAdd, sizeAdd * sizeof(float));
    cudaMallocManaged(&mxAdd, sizeAdd * sizeof(float));
    cudaMallocManaged(&myAdd, sizeAdd * sizeof(float));
}

void latticeAllDestructorAdapter(unsigned long long int *&Gmain, float *&Emain, float *&Mmain, 
    unsigned long long int *&Gresult, float *&Eresult, float *&Mresult,  
    float *&x, float *&y, float *&mx, float *&my, 
    float *&xMain, float *&yMain, float *&mxMain, float *&myMain, 
    float *&xAdd, float *&yAdd, float *&mxAdd, float *&myAdd)
{
    if(Mmain != nullptr)
    {
        cudaFree(Mmain);
        //printf("free Mmain\n");
    }
    if(Emain != nullptr)
    {
        cudaFree(Emain);
        //printf("free Emain\n");
    }
    if(Gmain != nullptr)
    {
        cudaFree(Gmain);
        //printf("free Gmain\n");
    }

    if(Mresult != nullptr)
    {
        cudaFree(Mresult);
        //printf("free Mresult\n");
    }
    if(Eresult != nullptr)
    {
        cudaFree(Eresult);
        //printf("free Eresult\n");
    }
    if(Gresult != nullptr)
    {
        cudaFree(Gresult);
        //printf("free Gresult\n");
    }

    if(my != nullptr)
    {
        cudaFree(my);
        //printf("free my\n");
    }
    if(mx != nullptr)
    {
        cudaFree(mx);
        //printf("free mx\n");
    }
    if(y != nullptr)
    {
        cudaFree(y);
        //printf("free y\n");
    }
    if(x != nullptr)
    {
        cudaFree(x);
        //printf("free x\n");
    }

    if(myMain != nullptr)
    {
        cudaFree(myMain);
        //printf("free myMain\n");
    }
    if(mxMain != nullptr)
    {
        cudaFree(mxMain);
        //printf("free mxMain\n");
    }
    if(yMain != nullptr)
    {
        cudaFree(yMain);
        //printf("free yMain\n");
    }
    if(xMain != nullptr)
    {
        cudaFree(xMain);
        //printf("free xMain\n");
    }

    if(myAdd != nullptr)
    {
        cudaFree(myAdd);
        //printf("free myAdd\n");
    }
    if(mxAdd != nullptr)
    {
        cudaFree(mxAdd);
        //printf("free mxAdd\n");
    }
    if(yAdd != nullptr)
    {
        cudaFree(yAdd);
        //printf("free yAdd\n");
    }
    if(xAdd != nullptr)
    {
        cudaFree(xAdd);
        //printf("free xAdd\n");
    }
}
