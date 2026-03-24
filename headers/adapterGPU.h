#ifndef adapterGPU_h
#define adapterGPU_h

#include "lattice.h"

void dosConstructorAdapter(unsigned long long *&G, float *&E, float *&M, 
    unsigned long long *&conf, int size);
void dosDestructorAdapter(unsigned long long *&G, float *&E, float *&M, 
    unsigned long long *&conf);
void latticeConstructorAdapter(float *&x, float *&y, float *&mx, float *&my, int size);
void latticeDestructorAdapter(float *&x, float *&y, float *&mx, float *&my);

#endif
