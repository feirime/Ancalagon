#ifndef adapterGPU_h
#define adapterGPU_h

#include "latticeFactory.h"

void latticeConstructorAdapter(long long int *&G, float *&E, int *&M, float *&x, float *&y, float *&mx, float *&my, int linearSize);
void latticeDestructorAdapter(long long int *&G, float *&E, int *&M, float *&x, float *&y, float *&mx, float *&my);
void calculateAdapter(long long int *&G, float *&E, int *&M);

#endif