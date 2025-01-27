#ifndef adapterGPU_h
#define adapterGPU_h

#include "latticeFactory.h"


void testAdapterGPU();
void latticeConstructor(long long int *G, float *E, int *M);
void latticeDestructor(long long int *G, float *E, int *M);

#endif