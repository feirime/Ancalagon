#ifndef adapterGPU_h
#define adapterGPU_h

#include "lattice.h"

void dosConstructorAdapter(unsigned long long *&G, float *&E, float *&M, unsigned long long *&conf, size_t size);
void dosDestructorAdapter(unsigned long long *&G, float *&E, float *&M, unsigned long long *&conf);
void latticeConstructorAdapter(float *&x, float *&y, float *&mx, float *&my, size_t size);
void latticeDestructorAdapter(float *&x, float *&y, float *&mx, float *&my);

void kernelElementaryAdapter(float *&x, float *&y, float *&mx, float *&my, int latticeSize,
    unsigned long long *&G, float *&E, float *&M, unsigned long long *&conf, size_t dosSize);
void kernelUnifyingAdapter(float *&xMain, float *&yMain, float *&mxMain, float *&myMain, size_t latticeMainSize,
    float *&xAdd, float *&yAdd, float *&mxAdd, float *&myAdd, size_t latticeAddSize,
    unsigned long long *&Gmain, float *&Emain, float *&Mmain, unsigned long long *&confMain, size_t dosMainSize,
    unsigned long long *&Gadd, float *&Eadd, float *&Madd, unsigned long long *&confAdd, size_t dosAddSize,
    unsigned long long *&Gresult, float *&Eresult, float *&Mresult, unsigned long long *&confResult, size_t dosResultSize);

#endif
