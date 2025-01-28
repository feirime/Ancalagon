#include "run.h"

void Run::run(int argc, char* argv[]) 
{
    arguments(argc, argv);
    Lattice* lattice = LatticeFactory::createLattice(latticeType);
    lattice->createLattice();
    lattice->calculate();
    lattice->print();
    delete lattice;
}

void Run::arguments(int argc, char* argv[])
{
    auto parser = argumentum::argument_parser{};
    auto params = parser.params();
    parser.config().program(argv[0]).description("Program for calculation density of states");
    params.add_parameter(linearSize, "-n", "--linearSize").nargs(1).required().metavar("linearSize").help("numbers of spins in one dimension");
    params.add_parameter(latticeType, "-t", "--latticeType").nargs(1).required().metavar("latticeType").help("type of lattice");
    params.add_parameter(lattice_read, "-r", "--read").absent(false).nargs(1).metavar("Read").help("Read J from file");
    params.add_parameter(read_pass, "--readpass").absent("data/").nargs(1).metavar("Read").help("Read J from file");
    params.add_parameter(device, "-d", "--device").absent("CPU").nargs(1).metavar("device").help("Calculate on device");
    auto res = parser.parse_args( argc, argv, 1 );
    if( !res )
        std::exit( 1 );
    lineaarLength = 1 << linearSize;
}
