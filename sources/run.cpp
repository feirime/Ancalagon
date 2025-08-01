#include <chrono>
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
        lattice->generateLattice();
    auto start_time = std::chrono::high_resolution_clock::now();
    if(calcStrategy == "unified")
    {
        lattice->init(iteractionRadius, accuracy, splitSeed);
        lattice->mapMaker();
        lattice->calculateMain();
        lattice->calculateAdd();
        lattice->calculateUnified();
        //lattice->compress();
        while(!lattice->isEnd())
        {
            lattice->layerPlusPlus();
            lattice->mapMaker();
            lattice->calculateAdd();
            lattice->calculateUnified();
        }
        //lattice->compress();
    }
    if(calcStrategy == "brutforce")
    {
        lattice->init(iteractionRadius, accuracy, splitSeed);
        lattice->dosMallocBrutforce();
        lattice->brutforce();
        lattice->compressRBTreeSE();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    int hours = elapsed_time.count() / 3600000;
    int minutes = (elapsed_time.count() % 3600000) / 60000;
    int seconds = (elapsed_time.count() % 60000) / 1000;
    int milliseconds = elapsed_time.count() % 1000;
    std::cout << "Elapsed time: " << hours << " hours, " << minutes << " minutes, " 
              << seconds << " seconds, " << milliseconds << " milliseconds" << std::endl;
    lattice->print();
    delete lattice;
}

void Run::arguments(int argc, char* argv[])
{
    auto parser = argumentum::argument_parser{};
    auto params = parser.params();
    parser.config().program(argv[0]).description("Program for calculation density of states");
    params.add_parameter(latticeSize, "-n", "--latticeSize").absent(0).nargs(1).metavar("latticeSize").help("numbers of spins in one dimension");
    //params.add_parameter(latticeType, "-t", "--latticeType").nargs(1).required().metavar("latticeType").help("type of lattice");
    params.add_parameter(lattice_read, "-r", "--read").absent(true).nargs(0).metavar("Read").help("Read J from file");
    params.add_parameter(readPass, "--readpass").absent("data/Read").nargs(1).metavar("Read").help("Read J from file");
    params.add_parameter(device, "-d", "--device").absent("cpu").nargs(1).metavar("device").help("Calculate on device");
    params.add_parameter(splitSeed, "-s", "--splitSeed").absent(1).nargs(1).metavar("splitSeed").help("seed of lattice spliting");
    params.add_parameter(iteractionRadius, "--radius").absent(2).nargs(1).metavar("iteractionRadius").help("radius of iteraction between spins");
    params.add_parameter(accuracy, "--accuracy").absent(1e-6).nargs(1).metavar("accuracy").help("accuracy of comparison E and M");
    params.add_parameter(calcStrategy, "--calcStrategy").absent("brutforce").nargs(1).metavar("calcStrategy").help("calculation strategy");
    auto res = parser.parse_args( argc, argv, 1 );
    if( !res )
        std::exit( 1 );
}

void Run::readIterator()
{
    numberOfFiles = 0;
    for(const auto& entry : std::filesystem::directory_iterator(readPass))
    {
        std::cout << entry.path() << '\n';
        numberOfFiles++;
    }
    fileName = new std::string[numberOfFiles];
    int i = 0;
    for(const auto& entry : std::filesystem::directory_iterator(readPass))
    {
        fileName[i] = entry.path();
        i++;
    }
}

Run::~Run()
{
    delete[] fileName;
}
