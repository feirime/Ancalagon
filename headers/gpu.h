#ifndef gpu_h
#define gpu_h
#include <algorithm>

int get_SP_cores(cudaDeviceProp devProp);
__global__ void ElementaryClalc(float *x, float *y, float *mx, float *my, int latticeSize,
    unsigned long long *G, float *E, float *M, unsigned long long *conf, size_t dosSize, size_t confSize, float iterationRadius);
__global__ void unifing(float *xMain, float *yMain, float *mxMain, float *myMain, size_t latticeMainSize,
    float *xAdd, float *yAdd, float *mxAdd, float *myAdd, size_t latticeAddSize,
    unsigned long long *Gmain, float *Emain, float *Mmain, unsigned long long *confMain, size_t dosMainSize, size_t confMainSize,
    unsigned long long *Gadd, float *Eadd, float *Madd, unsigned long long *confAdd, size_t dosAddSize, size_t confAddSize,
    unsigned long long *Gresult, float *Eresult, float *Mresult, unsigned long long *confResult, size_t dosResultSize, size_t confResultSize,
    float iteractionRadius);

#endif
