#ifndef adapterGPU_h
#define adapterGPU_h

#include "lattice.h"

void latticeConstructorDOSAdapter(unsigned long long int *&G, float *&E, float *&M, int size);
void latticeConstructorAdapter(float *&x, float *&y, float *&mx, float *&my, int size);
void latticeMainConstructorAdapter(float *&xMain, float *&yMain, float *&mxMain, float *&myMain, int size);
void latticeAddConstructorAdapter(float *&xAdd, float *&yAdd, float *&mxAdd, float *&myAdd, int size);
void latticeDestructorAdapter(unsigned long long int *&Gmain, float *&Emain, float *&Mmain, 
    unsigned long long int *&Gresult, float *&Eresult, float *&Mresult,  
    float *&x, float *&y, float *&mx, float *&my, 
    float *&xMain, float *&yMain, float *&mxMain, float *&myMain, 
    float *&xAdd, float *&yAdd, float *&mxAdd, float *&myAdd);

#endif
