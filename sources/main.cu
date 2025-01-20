#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <chrono>
#include <argumentum/argparse.h>
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

int main(int argc, char **argv)
{
    int n;
    int J_sum;
    int J_ferr;
    float P_plus;
    bool spin_glass_read;
    std::string read_pass;


    /************************************* arguments configuration ****************************************************/

    auto parser = argumentum::argument_parser{};
    auto params = parser.params();
    parser.config().program(argv[0]).description("Program for calculation density of states");
    params.add_parameter(n, "-n", "--number").nargs(1).required().metavar("N").help("numbers of spins in one dimension");
    params.add_parameter(J_sum, "-J").absent(10001).nargs(1).metavar("J")
                                     .help("Sum of exchange integrals");
    params.add_parameter(P_plus, "-p", "--p_plus").absent(-1).nargs(1).metavar("P+")
                                                  .help("Propability of positive exchange integrals");
    params.add_parameter(spin_glass_read, "-r", "--read").absent(false).nargs(0).metavar("Read").help("Read J from file");
    params.add_parameter(read_pass, "--readpass").absent("data/cell_read").nargs(0).metavar("Read")
                                                 .help("Read J from file");
    auto res = parser.parse_args( argc, argv, 1 );
    if( !res )
        return 1;
    if(J_sum != 10001 && P_plus !=-1)
    {
        std::cout << "Please, don't use both -J and -p keys\n";
        return 0;
    }
    if(J_sum == 10001 && P_plus == -1)
    {
        J_ferr = 2 * n * (n - 1);
    }
    else
    {
        if(P_plus != -1)
        {
            if(P_plus > 1 || P_plus < 0)
            {
                std::cout << "P_plus is out of bounds\n";
                return 0;
            }
            J_ferr = int(P_plus * 2 * n * (n - 1));
        }
        else
        {
            if(J_sum % 2 != 0)
            {
                std::cout << "Jsum can't be odd\n";
                return 0;
            }
            J_ferr = n * (n - 1) + J_sum / 2;
        }
    }
    if (J_ferr > 2 * (n * (n - 1)) || J_ferr < 0)
    {
        std::cout << "Jsum is out of bounds\n";
        return 0;
    }
    uint64_cu length = 1 << n;
    int sample_number = 0;

    /***************************************** GPU configuration ******************************************************/

    cudaDeviceProp dev{};
    cudaGetDeviceProperties(&dev, 0);
    static size_t block_dim = 512;
    static size_t grid_dim = get_SP_cores(dev);
    std::cout << "sp_cores: " << get_SP_cores(dev) << "\n";

    /***************************************** J_configurations *******************************************************/

    int J_h_host[(n - 1) * n];
    int J_v_host[n * (n - 1)];
    std::string *cell_name;
    if(spin_glass_read)
    {
        number_of_cell = 0;
        const std::filesystem::path read_dir{read_pass};
        for (auto const &dir_entry: std::filesystem::directory_iterator{read_dir})
            number_of_cell++;
        cell_name = new std::string [number_of_cell];
        int i = 0;
        for (auto const &dir_entry: std::filesystem::directory_iterator{read_dir})
        {
            std::string file_input_name{dir_entry.path().string()};
            cell_name[i] = file_input_name;
            i++;
        }
    }
    for(auto cell = 0; cell < number_of_cell; cell++)
    {
        std::string out_name_add;
        int sum_of_J = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        if(spin_glass_read)
        {
            J_read(n, J_h_host, J_v_host, cell_name[cell]);
            size_t str_start = cell_name[cell].find_last_of("_") + 1;
            size_t str_end = cell_name[cell].find_last_of(".");
            sample_number = std::stoi(cell_name[cell].substr(str_start, str_end - str_start));
            out_name_add = cell_name[cell];
            out_name_add.erase(0, out_name_add.find_last_of("/") + 1);
            J_write(n, J_h_host, J_v_host, sum_of_J, sample_number);
        }
        else
        {
            if(J_ferr > 2 * (n * (n - 1)))
            {
                std::cout << "to match J_ferr " << "\n";
                return 0;
            }
            J_glass_generator(n, J_h_host, J_v_host, J_ferr);
            sample_number = cell;
            J_write(n, J_h_host, J_v_host, sum_of_J, sample_number);
            out_name_add = std::to_string(n * n) + "_J" + std::to_string(sum_of_J) + '_' 
                                    + std::to_string(sample_number) + ".dat";
        }
        printf("J_sum = %d\n", sum_of_J);
        std::ofstream log("data/logs/log_" + out_name_add);
        log.close();
        CUDA_CHECK_ERROR(cudaMemcpyToSymbol(J_horizontal, &J_h_host, n * (n - 1) * sizeof(int)))
        CUDA_CHECK_ERROR(cudaMemcpyToSymbol(J_vertical, &J_v_host, n * (n - 1) * sizeof(int)))
        log.open("data/logs/log_" + out_name_add, std::ios::app);
        log << "sp_cores: " << get_SP_cores(dev) << "\n";
        log << "sum of J = " << sum_of_J << "\n";
        log.close();

        /***************************************** Base creations *********************************************************/

        uint16_cu *prime_set;
        CUDA_CHECK_ERROR(cudaMallocManaged(&prime_set, (prime_n - 1) * sizeof(uint16_cu)))
        prime_set[0] = 997;
        prime_set[1] = 991;
        prime_set[2] = 983;
        prime_set[3] = 977;
        prime_set[4] = 971;
        prime_set[5] = 967;
        prime_set[6] = 953;
        prime_set[7] = 947;
        prime_set[8] = 941;
        prime_set[9] = 937;
        prime_set[10] = 929;
        prime_set[11] = 919;
        prime_set[12] = 911;
        prime_set[13] = 907;
        prime_set[14] = 887;
        prime_set[15] = 883;
        prime_set[16] = 881;
        prime_set[17] = 877;
        prime_set[18] = 863;
        prime_set[19] = 859;
        int8_cu *S;
        CUDA_CHECK_ERROR(cudaMallocManaged(&S, length * n * sizeof(int8_cu)))
        thread_init(n, S);
        uint64_cu *EMC_original, *EMC_even, *EMC_odd;
        int column = 0;
        int E_max_o = 2 * (n - 1) + 1;
        int M_max_o = 2 * n + 1;
        int E_max_even = 2 * ((n - 1) * (column + 1) + column * n) + 1;
        int M_max_even = 2 * n * (column + 1) + 1;
        size_t EMC_size = E_max_o * M_max_o * length * prime_n * sizeof(uint64_cu);
        CUDA_CHECK_ERROR(cudaMallocManaged(&EMC_even, EMC_size))
        initializer<<<grid_dim, block_dim>>>(n, EMC_even, E_max_even, M_max_even);
        generator_EMC<<<grid_dim, block_dim>>>(n, EMC_even, S,
                                               E_max_even, M_max_even, column);
        mod<<<grid_dim, block_dim>>>(n, EMC_even, E_max_even, M_max_even, prime_set);
        cudaDeviceSynchronize();
        column++;

        /******************************************* Unifying *************************************************************/

        CUDA_CHECK_ERROR(cudaMallocManaged(&EMC_original, EMC_size))
        initializer<<<grid_dim, block_dim>>>(n, EMC_original, E_max_o, M_max_o);
        generator_EMC<<<grid_dim, block_dim>>>(n, EMC_original, S,
                                               E_max_o, M_max_o, column);
        mod<<<grid_dim, block_dim>>>(n, EMC_original, E_max_o, M_max_o, prime_set);
        cudaDeviceSynchronize();
        int E_max_odd = 2 * ((n - 1) * (column + 1) + column * n) + 1;
        int M_max_odd = 2 * n * (column + 1) + 1;
        EMC_size = E_max_odd * M_max_odd * length * prime_n * sizeof(uint64_cu);
        CUDA_CHECK_ERROR(cudaMallocManaged(&EMC_odd, EMC_size))
        initializer<<<grid_dim, block_dim>>>(n, EMC_odd, E_max_odd, M_max_odd);
        unifying<<<grid_dim, block_dim>>>(n, EMC_even, E_max_even, M_max_even,
                                          EMC_original, E_max_o, M_max_o,
                                          EMC_odd, E_max_odd, M_max_odd,
                                          S, column - 1);
        mod<<<grid_dim, block_dim>>>(n, EMC_odd, E_max_odd, M_max_odd, prime_set);
        cudaDeviceSynchronize();
        log.open("data/logs/log_" + out_name_add, std::ios::app);
        std::cout << "size of " << column + 1 << " EMC = " << EMC_size / (1 << 10) << " kb \n";
        log << "size of " << column + 1 << " EMC = " << EMC_size / (1 << 10) << " kb \n";
        log.close();
        column++;
        while (column < (n - 1))
        {
            if (column % 2 > 0)
            {
                CUDA_CHECK_ERROR(cudaFree(EMC_odd))
                initializer<<<grid_dim, block_dim>>>(n, EMC_original, E_max_o, M_max_o);
                generator_EMC<<<grid_dim, block_dim>>>(n, EMC_original, S,
                                                       E_max_o, M_max_o, column);
                mod<<<grid_dim, block_dim>>>(n, EMC_original, E_max_o, M_max_o, prime_set);
                E_max_odd = 2 * ((n - 1) * (column + 1) + column * n) + 1;
                M_max_odd = 2 * n * (column + 1) + 1;
                EMC_size = E_max_odd * M_max_odd * length * prime_n * sizeof(uint64_cu);
                CUDA_CHECK_ERROR(cudaMallocManaged(&EMC_odd, EMC_size))
                initializer<<<grid_dim, block_dim>>>(n, EMC_odd, E_max_odd, M_max_odd);
                unifying<<<grid_dim, block_dim>>>(n, EMC_even, E_max_even, M_max_even,
                                                  EMC_original, E_max_o, M_max_o,
                                                  EMC_odd, E_max_odd, M_max_odd,
                                                  S, column - 1);
                mod<<<grid_dim, block_dim>>>(n, EMC_odd, E_max_odd, M_max_odd, prime_set);
                cudaDeviceSynchronize();
                std::cout << "size of " << column + 1 << " EMC = " << EMC_size / (1 << 10) << " kb \n";
                log.open("data/logs/log_" + out_name_add, std::ios::app);
                log << "size of " << column + 1 << " EMC = " << EMC_size / (1 << 10) << " kb \n";
                log.close();
                column++;
            }
            else
            {
                CUDA_CHECK_ERROR(cudaFree(EMC_even))
                initializer<<<grid_dim, block_dim>>>(n, EMC_original, E_max_o, M_max_o);
                generator_EMC<<<grid_dim, block_dim>>>(n, EMC_original, S,
                                                       E_max_o, M_max_o, column);
                mod<<<grid_dim, block_dim>>>(n, EMC_original, E_max_o, M_max_o, prime_set);
                E_max_even = 2 * ((n - 1) * (column + 1) + column * n) + 1;
                M_max_even = 2 * n * (column + 1) + 1;
                EMC_size = E_max_even * M_max_even * length * prime_n * sizeof(uint64_cu);
                CUDA_CHECK_ERROR(cudaMallocManaged(&EMC_even, EMC_size))
                initializer<<<grid_dim, block_dim>>>(n, EMC_even, E_max_even, M_max_even);
                unifying<<<grid_dim, block_dim>>>(n, EMC_odd, E_max_odd, M_max_odd,
                                                  EMC_original, E_max_o, M_max_o,
                                                  EMC_even, E_max_even, M_max_even,
                                                  S, column - 1);
                mod<<<grid_dim, block_dim>>>(n, EMC_even, E_max_even, M_max_even, prime_set);
                cudaDeviceSynchronize();
                std::cout << "size of " << column + 1 << " EMC = " << EMC_size / (1 << 10) << " kb \n";
                log.open("data/logs/log_" + out_name_add, std::ios::app);
                log << "size of " << column + 1 << " EMC = " << EMC_size / (1 << 10) << " kb \n";
                log.close();
                column++;
            }
        }
        if (column % 2 > 0)
        {
            CUDA_CHECK_ERROR(cudaFree(EMC_odd))
            initializer<<<grid_dim, block_dim>>>(n, EMC_original, E_max_o, M_max_o);
            generator_EMC<<<grid_dim, block_dim>>>(n, EMC_original, S,
                                                   E_max_o, M_max_o, column);
            mod<<<grid_dim, block_dim>>>(n, EMC_original, E_max_o, M_max_o, prime_set);
            E_max_odd = 2 * ((n - 1) * (column + 1) + column * n) + 1;
            M_max_odd = 2 * n * (column + 1) + 1;
            EMC_size = E_max_odd * M_max_odd * prime_n * sizeof(uint64_cu);
            CUDA_CHECK_ERROR(cudaMallocManaged(&EMC_odd, EMC_size))
            last_initializer<<<grid_dim, block_dim>>>(EMC_odd, E_max_odd, M_max_odd);
            last_unifying<<<grid_dim, block_dim>>>(n, EMC_even, E_max_even, M_max_even,
                                                   EMC_original, E_max_o, M_max_o,
                                                   EMC_odd, E_max_odd, M_max_odd,
                                                   S, column - 1);
            last_mod<<<grid_dim, block_dim>>>(EMC_odd, E_max_odd, M_max_odd, prime_set);
            cudaDeviceSynchronize();
            std::cout << "size of last EMC = " << EMC_size / (1 << 10) << " kb \n";
            log.open("data/logs/log_" + out_name_add, std::ios::app);
            log << "size of last EMC = " << EMC_size / (1 << 10) << " kb \n";
            log.close();
            out(n, EMC_odd, E_max_odd, M_max_odd, prime_set, sum_of_J, sample_number, out_name_add);
        }
        else
        {
            CUDA_CHECK_ERROR(cudaFree(EMC_even))
            initializer<<<grid_dim, block_dim>>>(n, EMC_original, E_max_o, M_max_o);
            generator_EMC<<<grid_dim, block_dim>>>(n, EMC_original, S,
                                                   E_max_o, M_max_o, column);
            mod<<<grid_dim, block_dim>>>(n, EMC_original, E_max_o, M_max_o, prime_set);
            E_max_even = 2 * ((n - 1) * (column + 1) + column * n) + 1;
            M_max_even = 2 * n * (column + 1) + 1;
            EMC_size = E_max_even * M_max_even * prime_n * sizeof(uint64_cu);
            CUDA_CHECK_ERROR(cudaMallocManaged(&EMC_even, EMC_size))
            last_initializer<<<grid_dim, block_dim>>>(EMC_even, E_max_even, M_max_even);
            last_unifying<<<grid_dim, block_dim>>>(n, EMC_odd, E_max_odd, M_max_odd,
                                                   EMC_original, E_max_o, M_max_o,
                                                   EMC_even, E_max_even, M_max_even,
                                                   S, column - 1);
            last_mod<<<grid_dim, block_dim>>>(EMC_even, E_max_even, M_max_even,
                                              prime_set);
            cudaDeviceSynchronize();
            std::cout << "size of last EMC = " << EMC_size / (1 << 10) << " kb \n";
            log.open("data/logs/log_" + out_name_add, std::ios::app);
            log << "size of last EMC = " << EMC_size / (1 << 10) << " kb \n";
            log.close();
            out(n, EMC_even, E_max_even, M_max_even, prime_set, sum_of_J, sample_number, out_name_add);
        }
        cudaDeviceSynchronize();

        /******************************************** Memory cleaning *****************************************************/

        CUDA_CHECK_ERROR(cudaFree(S))
        CUDA_CHECK_ERROR(cudaFree(EMC_original))
        CUDA_CHECK_ERROR(cudaFree(prime_set))
        CUDA_CHECK_ERROR(cudaFree(EMC_even))
        CUDA_CHECK_ERROR(cudaFree(EMC_odd))
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "working time is " << time / 3600000 << " h " << (time % 3600000) / 60000 << " m "
                  << ((time % 3600000) % 60000) / 1000 << " s " << ((time % 3600000) % 60000) % 1000 << " ms \n";
        log.open("data/logs/log_" + out_name_add, std::ios::app);
        log << "working time is " << time / 3600000 << " h " << (time % 3600000) / 60000 << " m "
            << ((time % 3600000) % 60000) / 1000 << " s " << ((time % 3600000) % 60000) % 1000 << " ms \n";
        log.close();
    }
}
