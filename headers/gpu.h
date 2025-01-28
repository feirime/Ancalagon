#ifndef gpu_h
#define gpu_h

int get_SP_cores(cudaDeviceProp devProp);
//__global__ void unifingSquare(int rightLayer, Cell *left, int leftSize, Cell *right, int rightSize, Cell *result, float *JMap);
__global__ void test();

#endif
