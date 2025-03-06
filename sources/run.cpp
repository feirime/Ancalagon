#include "run.h"

void Run::run(int argc, char* argv[]) 
{
    arguments(argc, argv);
    Lattice* lattice = LatticeFactory::createLattice(device);
    std::cout << "Lattice created by " << device << '\n';
    if(lattice_read)
    {
        readIterator();
        latticeSize = lattice->read(fileName[0]); //TODO реализовать итерацию по всем решеткам
    }
    else
        lattice->generateLattice(latticeSize);
    lattice->createDOS(latticeSize);
    lattice->calculate(latticeSize, splitSeed);
    lattice->print();
    delete lattice;
}

void Run::arguments(int argc, char* argv[])
{
    auto parser = argumentum::argument_parser{};
    auto params = parser.params();
    parser.config().program(argv[0]).description("Program for calculation density of states");
    params.add_parameter(latticeSize, "-n", "--latticeSize").nargs(1).required().metavar("latticeSize").help("numbers of spins in one dimension");
    //params.add_parameter(latticeType, "-t", "--latticeType").nargs(1).required().metavar("latticeType").help("type of lattice");
    params.add_parameter(lattice_read, "-r", "--read").absent(false).nargs(0).metavar("Read").help("Read J from file");
    params.add_parameter(readPass, "--readpass").absent("data/Read").nargs(1).metavar("Read").help("Read J from file");
    params.add_parameter(device, "-d", "--device").absent("CPU").nargs(1).metavar("device").help("Calculate on device");
    params.add_parameter(splitSeed, "-s", "--splitSeed").absent(0.1).nargs(1).metavar("splitSeed").help("seed of lattice spliting");
    auto res = parser.parse_args( argc, argv, 1 );
    if( !res )
        std::exit( 1 );
    lineaarLength = 1 << latticeSize;
}

void Run::readIterator()
{
    numberOfFiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(readPass))
    {
        std::cout << entry.path() << '\n';
        numberOfFiles++;
    }
    fileName = new std::string[numberOfFiles];
    int i = 0;
    for (const auto& entry : std::filesystem::directory_iterator(readPass))
    {
        fileName[i] = entry.path();
        i++;
    }
}

Run::~Run() 
{
    delete[] fileName;
}
