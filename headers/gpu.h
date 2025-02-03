#ifndef gpu_h
#define gpu_h
#include <algorithm>

int get_SP_cores(cudaDeviceProp devProp);
__global__ void test();
void mapMaker(float *&x, float *&y, float *&mx, float *&my, int latticeSize, float seed);
__global__ void calculate(long long int *G, float *E, int *M, float *&x, float *&y, float *&mx, float *&my, int latticeSize);


#endif
