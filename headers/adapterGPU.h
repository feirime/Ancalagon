#ifndef adapterGPU_h
#define adapterGPU_h

#include "latticeFactory.h"

void latticeConstructorAdapter(long long int *&G, float *&E, int *&M);
void latticeDestructorAdapter(long long int *&G, float *&E, int *&M);
void calculateAdapter(long long int *&G, float *&E, int *&M);

#endif