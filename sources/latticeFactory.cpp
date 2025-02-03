#include <fstream>
#include <iostream>
#include <algorithm>
#include "latticeFactory.h"

Lattice* LatticeFactory::createLattice(std::string device) 
{
    if (device == "GPU") 
    {
        return new LatticeGPU();
    }
    else if (device == "CPU") 
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

LatticeGPU::~LatticeGPU() 
{
    latticeDestructorAdapter(G, E, M, x, y, mx, my);
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
    fileContents.close();
    std::ifstream file(fileName);
    latticeConstructorAdapter(x, y, mx, my, count / 4);
    double temp = 0;  
    for(auto i = 0; i < count / 4; i++)
    {
        file >> x[i];
        file >> y[i];
        file >> mx[i];
        file >> my[i];
    }
    file.close();
    return count / 4;
}

void generateLattice(int latticeSize) 
{}

void LatticeGPU::createDOS(int latticeSize) 
{
    latticeConstructorDOSAdapter(G, E, M);
};

void LatticeGPU::calculate(int latticeSize, float splitSeed)
{
    if(E != nullptr && G != nullptr && M != nullptr)
    {
        calculateAdapter(G, E, M, x, y, mx, my, latticeSize, splitSeed);
    }
}

void Lattice::print()
{}
