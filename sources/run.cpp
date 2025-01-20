#include "run.h"

void Run::run(int argc, char* argv[]) 
{
    arguments(argc, argv);
}

int Run::arguments(int argc, char* argv[])
{
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
}