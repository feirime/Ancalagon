#ifndef adapterGPU_h
#define adapterGPU_h

#include "lattice.h"

void latticeConstructorDOSAdapter(unsigned long long int *&G, float *&E, float *&M, int size);
void latticeConstructorAdapter(float *&x, float *&y, float *&mx, float *&my, int size);
void latticeDestructorAdapter(unsigned long long int *&Gmain, float *&Emain, float *&Mmain, 
    unsigned long long int *&Gresult, float *&Eresult, float *&Mresult,  float *&x, float *&y, float *&mx, float *&my);
void calculateAdapter(unsigned long long int *&G, float *&E, int *&M,
     float *&x, float *&y, float *&mx, float *&my, int latticeSize, float splitSeed);

#endif
