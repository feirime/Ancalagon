#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <chrono>
#include "cpu.h"
#include "gpu.h"
#include "chinese_remainder_theorem.h"

int const prime_n = 14;
int const samples_per_J = 1;
int number_of_cell = samples_per_J;

__device__ int const prime_n_dev = prime_n;
__constant__ int J_horizontal[360];
__constant__ int J_vertical[360];

#define CUDA_CHECK_ERROR(err)           \
if ((err) != cudaSuccess)               \
{                                       \
    log.open("data/logs/log" + out_name_add, std::ios::app);\
    log << "Cuda error: \n";          \
    log << cudaGetErrorString(err) << "\n";                                               \
    log.close();                                    \
    printf("Cuda error: %s\n", cudaGetErrorString(err));    \
    printf("Error in file: %s, line: %i\n", __FILE__, __LINE__); \
    return 1;\
}

int main(int argc, char* argv[])
{}
