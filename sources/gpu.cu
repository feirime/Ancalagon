
#include <iostream>
#include "gpu.h"

int get_SP_cores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

__global__ void generator_EMC(int n_dev, uint64_cu *EMC, const int8_cu *S, int E_max, int M_max, int m)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_cu length_dev = 1 << n_dev;
    int E_add = (E_max - 1) / 2;
    int M_add = (M_max - 1) / 2;
    for(auto i = x; i < length_dev; i += blockDim.x * gridDim.x)
    {
        int E = 0;
        int M = 0;
        for(auto j = 0; j < (n_dev - 1); ++j)
        {
            E += -J_vertical[j * n_dev + m] * S[i * n_dev + j] * S[i * n_dev + j + 1];
            M += S[i * n_dev + j];
        }
        M += S[i * n_dev + n_dev - 1];
        E += E_add;
        M += M_add;
        atomicAdd(&EMC[i * E_max * M_max * prime_n_dev +
                               E * M_max * prime_n_dev + M * prime_n_dev], 1);
        for(auto j = 1; j < prime_n_dev; ++j)
        {
            atomicAdd(&EMC[i * E_max * M_max * prime_n_dev +
            E * M_max * prime_n_dev + M * prime_n_dev + j], 1);
        }
    }
}

__global__ void unifying(int n_dev, uint64_cu *EMC_l, int E_max_l, int M_max_l,
                         const uint64_cu *EMC_r, int E_max_r, int M_max_r,
                         uint64_cu *EMC_created, int E_max_created, int M_max_created,
                         const int8_cu *S, int m)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_cu length_dev = 1 << n_dev;
    int E_add_l = (E_max_l - 1) / 2;
    int M_add_l = (M_max_l - 1) / 2;
    int E_add_r = (E_max_r - 1) / 2;
    int M_add_r = (M_max_r - 1) / 2;
    int E_add_created = (E_max_created - 1) / 2;
    int M_add_created = (M_max_created - 1) / 2;
    for(auto conf_l = x; conf_l < length_dev; conf_l += blockDim.x * gridDim.x)
        for(auto E_l = 0; E_l < E_max_l; ++E_l)
            for(auto M_l = 0; M_l < M_max_l; ++M_l)
                for(auto conf_r = 0; conf_r < length_dev; ++conf_r)
                    for(auto E_r = 0; E_r < E_max_r; ++E_r)
                        for(auto M_r = 0; M_r < M_max_r; ++M_r)
                        {
                            if(EMC_l[conf_l * E_max_l * M_max_l * prime_n_dev +
                            E_l * M_max_l * prime_n_dev + M_l * prime_n_dev] > 0
                               && EMC_r[conf_r * E_max_r * M_max_r * prime_n_dev +
                               E_r * M_max_r * prime_n_dev + M_r * prime_n_dev] > 0)
                            {
                                int E = 0;
                                int M = 0;
                                for (auto j = 0; j < n_dev; ++j)
                                {
                                    E += -J_horizontal[j * (n_dev - 1) + m] * S[conf_l * n_dev + j] *
                                            S[conf_r * n_dev + j];
                                }
                                E += E_l - E_add_l + E_r - E_add_r + E_add_created;
                                M += M_l - M_add_l + M_r - M_add_r + M_add_created;
                                atomicAdd(&EMC_created[conf_r * E_max_created * M_max_created * prime_n_dev +
                                                       E * M_max_created * prime_n_dev + M * prime_n_dev], 1);
                                for(auto j = 1; j < prime_n_dev; ++j)
                                {
                                    atomicAdd(&EMC_created[conf_r * E_max_created * M_max_created * prime_n_dev +
                                                           E * M_max_created * prime_n_dev + M * prime_n_dev + j],
                                              EMC_l[conf_l * E_max_l * M_max_l * prime_n_dev +
                                              E_l * M_max_l * prime_n_dev + M_l * prime_n_dev + j]);
                                }
                            }
                        }
}

__global__ void last_unifying(int n_dev, uint64_cu *EMC_l, int E_max_l, int M_max_l,
                              const uint64_cu *EMC_r, int E_max_r, int M_max_r,
                              uint64_cu *EMC_created, int E_max_created, int M_max_created,
                              const int8_cu *S, int m)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_cu length_dev = 1 << n_dev;
    int E_add_l = (E_max_l - 1) / 2;
    int M_add_l = (M_max_l - 1) / 2;
    int E_add_r = (E_max_r - 1) / 2;
    int M_add_r = (M_max_r - 1) / 2;
    int E_add_created = (E_max_created - 1) / 2;
    int M_add_created = (M_max_created - 1) / 2;
    for(auto conf_l = x; conf_l < length_dev; conf_l += blockDim.x * gridDim.x)
        for(auto E_l = 0; E_l < E_max_l; ++E_l)
            for(auto M_l = 0; M_l < M_max_l; ++M_l)
                for(auto conf_r = 0; conf_r < length_dev; ++conf_r)
                    for(auto E_r = 0; E_r < E_max_r; ++E_r)
                        for(auto M_r = 0; M_r < M_max_r; ++M_r)
                        {
                            if(EMC_l[conf_l * E_max_l * M_max_l * prime_n_dev +
                            E_l * M_max_l * prime_n_dev + M_l * prime_n_dev] > 0
                               && EMC_r[conf_r * E_max_r * M_max_r * prime_n_dev +
                               E_r * M_max_r * prime_n_dev + M_r * prime_n_dev] > 0)
                            {
                                int E = 0;
                                int M = 0;
                                for (auto j = 0; j < n_dev; ++j)
                                {
                                    E += -J_horizontal[j * (n_dev - 1) + m] *
                                            S[conf_l * n_dev + j] * S[conf_r * n_dev + j];
                                }
                                E += E_l - E_add_l + E_r - E_add_r + E_add_created;
                                M += M_l - M_add_l + M_r - M_add_r + M_add_created;
                                atomicAdd(&EMC_created[E * M_max_created * prime_n_dev + M * prime_n_dev], 1);
                                for(auto j = 1; j < prime_n_dev; ++j)
                                {
                                    atomicAdd(&EMC_created[E * M_max_created * prime_n_dev + M * prime_n_dev + j],
                                              EMC_l[conf_l * E_max_l * M_max_l * prime_n_dev +
                                              E_l * M_max_l * prime_n_dev + M_l * prime_n_dev + j]);
                                }
                            }
                        }
}

__global__ void initializer(int n_dev, uint64_cu *EMC, int E_max, int M_max)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_cu length_dev = 1 << n_dev;
    for(auto i = x; i < (E_max * M_max * length_dev * prime_n_dev); i += blockDim.x * gridDim.x)
    {
        EMC[i] = 0;
    }
}

__global__ void last_initializer(uint64_cu *EMC, int E_max, int M_max)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    for(auto i = x; i < (E_max * M_max * prime_n_dev); i += blockDim.x * gridDim.x)
    {
        EMC[i] = 0;
    }
}

__global__ void mod(int n_dev, uint64_cu *EMC, int E_max, int M_max, const uint16_cu *prime_set)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_cu length_dev = 1 << n_dev;
    for(auto conf = x; conf < length_dev; conf += blockDim.x * gridDim.x)
        for(auto E = 0; E < E_max; ++E)
            for(auto M = 0; M < M_max; ++M)
                if(EMC[conf * E_max * M_max * prime_n_dev + E * M_max * prime_n_dev + M * prime_n_dev] > 0)
                    for(auto i = 1; i < prime_n_dev; ++i)
                    {
                        EMC[conf * E_max * M_max * prime_n_dev +
                        E * M_max * prime_n_dev + M * prime_n_dev + i] %= prime_set[i - 1];
                    }
}

__global__ void last_mod(uint64_cu *EMC, int E_max, int M_max, const uint16_cu *prime_set)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
        for(auto E = x; E < E_max; E += blockDim.x * gridDim.x)
            for(auto M = 0; M < M_max; ++M)
                if(EMC[E * M_max * prime_n_dev + M * prime_n_dev] > 0)
                    for(auto i = 1; i < prime_n_dev; ++i)
                    {
                        EMC[E * M_max * prime_n_dev + M * prime_n_dev + i] %= prime_set[i - 1];
                    }
}