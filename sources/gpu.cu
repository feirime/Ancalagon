
#include <iostream>
#include "gpu.h"

__global__ void unifingSquare(int rightLayer, Cell *left, int leftSize, Cell *right, int rightSize, Cell *result, float *JMap) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    for(int i = 0; i < leftSize; i++)
    {
        for(int j = 0; j < rightSize; j++)
        {
            result[j].M = left[i].M + right[j].M;
            for(int bound = 0; bound < 1 + rightLayer * 3; bound++)
            {
                result[j].E = left[i].E + right[j].E + JMap[i];
            }
            result[j].G = left[i].G;
        }
    }
}
