#ifndef adapterGPU_h
#define adapterGPU_h

#include "latticeFactory.h"

void latticeConstructorDOSAdapter(long long int *&G, float *&E, int *&M);
void latticeConstructorAdapter(double *&x, double *&y, double *&mx, double *&my, int size);
void latticeDestructorAdapter(long long int *&G, float *&E, int *&M, double *&x, double *&y, double *&mx, double *&my);
void calculateAdapter(long long int *&G, float *&E, int *&M, double *&x, double *&y, double *&mx, double *&my);

#endif
