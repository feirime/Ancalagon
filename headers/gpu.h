#ifndef gpu_h
#define gpu_h

struct Cell 
{
    long long int G;
    float E;
    int M;
};

__global__ void unifingSquare(int rightLayer, Cell *left, int leftSize, Cell *right, int rightSize, Cell *result, float *JMap);
__global__ void test();

#endif
