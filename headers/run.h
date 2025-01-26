#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <chrono>
#include <argumentum/argparse.h>
#include "adapterGPU.h"

class Run
{
private:
    int linearSize;
    long long int lineaarLength;
    bool sell_read;
    std::string read_pass;
    std::string cellType;
public:
    void run(int argc, char* argv[]);
    void arguments(int argc, char* argv[]);
    void out(int n, int* EMC, int* E_max, int* M_max, int* prime_set, int sum_of_J, int sample_number, std::string out_name_add);
};
