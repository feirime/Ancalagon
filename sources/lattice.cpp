#include "lattice.h"

Lattice* LatticeFactory::createLattice(std::string device) 
{
    if (device == "gpu") 
    {
        return new LatticeGPU();
    }
    else if (device == "cpu") 
    {
        return new LatticeCPU();
    }
    else if (device == "gibrid") 
    {
        return new LatticeGibrid();
    }
    else 
    {
        std::exit(1);
    }
}

Lattice::~Lattice() 
{
    std::cout << "Lattice destructor\n";
}

int Lattice::read(std::string fileName) 
{
    std::ifstream fileContents(fileName);
    if(!fileContents.is_open())
    {
        std::cout << "Where is File, Lebovski!?\n";
        std::exit(1);
    }
    std::string contents((std::istreambuf_iterator<char>(fileContents)), std::istreambuf_iterator<char>());
    int count = std::count(contents.begin(), contents.end(), ' ') + std::count(contents.begin(), contents.end(), '\n') + 1;
    if(count == 0)
    {
        std::cout << "File is empty\n";
        std::exit(1);
    }
    latticeSize = count / 4;
    std::ifstream file(fileName);
    printf("count = %d, latticeSize = %d\n", count, latticeSize);
    fileContents.close();
    latticeMalloc();
    double temp = 0;
    printf("fileName = %s\n", fileName.c_str());
    for(auto i = 4; i < latticeSize + 1; i++)
    {
        file >> x[i];
        file >> y[i];
        file >> mx[i];
        file >> my[i];
    }
    file.close();
    return latticeSize;
}

void Lattice::print()
{
    std::cout << "x: " << "y " << "mx " << "my " << '\n';
    for(int i = 0; i < latticeSize; i++)
    {
        std::cout << x[i] << " " << y[i] << " " << mx[i] << " " << my[i] << '\n';
    }
}

