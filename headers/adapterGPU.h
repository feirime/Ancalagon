#ifndef adapterGPU_h
#define adapterGPU_h

#include "lattice.h"

void latticeConstructorDOSAdapter(long long int *&G, float *&E, int *&M);
void latticeConstructorAdapter(float *&x, float *&y, float *&mx, float *&my, int size);
void latticeDestructorAdapter(long long int *&G, float *&E, int *&M, float *&x, float *&y, float *&mx, float *&my);
void calculateAdapter(long long int *&G, float *&E, int *&M, float *&x, float *&y, float *&mx, float *&my, int latticeSize, float splitSeed);

#endif
