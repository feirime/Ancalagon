#include <fstream>
#include <iostream>
#include <algorithm>
#include <regex>
#include "latticeFactory.h"

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
    printf("count = %d, latticeSize = %d\n", count, latticeSize);
    fileContents.close();
    std::ifstream file(fileName);
    latticeMalloc();
    double temp = 0;  
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

LatticeGPU::~LatticeGPU() 
{
    std::cout << "GPU destructor\n";
    latticeDestructorAdapter(G, E, M, x, y, mx, my);
}

void LatticeGPU::createDOS(int latticeSize) 
{
    latticeConstructorDOSAdapter(G, E, M);
    latticeConstructorAdapter(x, y, mx, my, latticeSize);
};

void LatticeGPU::calculate(int latticeSize, float splitSeed)
{
    if(E != nullptr && G != nullptr && M != nullptr)
    {
        calculateAdapter(G, E, M, x, y, mx, my, latticeSize, splitSeed);
    }
}

LatticeCPU::~LatticeCPU() 
{
    std::cout << "CPU destructor\n";
    delete [] G;
    delete [] E;
    delete [] M;
    delete [] x;
    delete [] y;
    delete [] mx;
    delete [] my;
}

void LatticeCPU::latticeMalloc()
{
    x = new float[latticeSize];
    y = new float[latticeSize];
    mx = new float[latticeSize];
    my = new float[latticeSize];
}
