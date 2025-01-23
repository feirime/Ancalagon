#include "run.h"

void Run::run(int argc, char* argv[]) 
{
    arguments(argc, argv);
    testAdapter();
}

int Run::arguments(int argc, char* argv[])
{
    auto parser = argumentum::argument_parser{};
    auto params = parser.params();
    parser.config().program(argv[0]).description("Program for calculation density of states");
    params.add_parameter(linearSize, "-n", "--linearSize").nargs(1).required().metavar("N").help("numbers of spins in one dimension");
    params.add_parameter(sell_read, "-r", "--read").absent(false).nargs(0).metavar("Read").help("Read J from file");
    params.add_parameter(read_pass, "--readpass").absent("data/cell_read").nargs(0).metavar("Read")
                                                 .help("Read J from file");
    auto res = parser.parse_args( argc, argv, 1 );
    if( !res )
        return 1;
    lineaarLength = 1 << linearSize;
    return 0;
}

void Run::out(int n, int* EMC, int* E_max, int* M_max, int* prime_set, int sum_of_J, int sample_number, std::string out_name_add)
{}