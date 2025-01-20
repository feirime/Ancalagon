#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <chrono>
#include <argumentum/argparse.h>
#include "cpu.h"
#include "gpu.h"
#include "chinese_remainder_theorem.h"

class Run
{
private:
    int const prime_n = 14;
    int n;
    int J_sum;
    int J_ferr;
    float P_plus;
    bool spin_glass_read;
    std::string read_pass;
public:
    void run(int argc, char* argv[]);
    int arguments(int argc, char* argv[]);
    void out(int n, int* EMC, int* E_max, int* M_max, int* prime_set, int sum_of_J, int sample_number, std::string out_name_add);
};