#ifndef gpu_h
#define gpu_h

typedef unsigned long long int uint64_cu;
typedef unsigned int uint16_cu;
typedef signed char int8_cu;

extern __device__ int const prime_n_dev;
extern __constant__ int J_horizontal[];
extern __constant__ int J_vertical[];

int get_SP_cores(cudaDeviceProp devProp);
__global__ void generator_EMC(int n_dev, uint64_cu *EMC, const int8_cu *S, int E_max, int M_max, int m);
__global__ void unifying(int n_dev, uint64_cu *EMC_l, int E_max_l, int M_max_l,
                         const uint64_cu *EMC_r, int E_max_r, int M_max_r,
                         uint64_cu *EMC_created, int E_max_created, int M_max_created,
                         const int8_cu *S, int m);
__global__ void last_unifying(int n_dev, uint64_cu *EMC_l, int E_max_l, int M_max_l,
                              const uint64_cu *EMC_r, int E_max_r, int M_max_r,
                              uint64_cu *EMC_created, int E_max_created, int M_max_created,
                              const int8_cu *S, int m);
__global__ void initializer(int n_dev, uint64_cu *EMC, int E_max, int M_max);
__global__ void last_initializer(uint64_cu *EMC, int E_max, int M_max);
__global__ void mod(int n_dev, uint64_cu *EMC, int E_max, int M_max, const uint16_cu *prime_set);
__global__ void last_mod(uint64_cu *EMC, int E_max, int M_max, const uint16_cu *prime_set);

#endif
