#ifndef gpu_h
#define gpu_h
#include <algorithm>

int get_SP_cores(cudaDeviceProp devProp);
__global__ void test();
void mapMaker(double *x, double *y, double *mx, double *my, int latticeSize);
__global__ void calculate(long long int *G, float *E, int *M, double *x, double *y, double *mx, double *my, int latticeSize);


#endif
