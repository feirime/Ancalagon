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

void Lattice::read(std::string fileName) 
{
    std::ifstream file(fileName);
    if(!file.is_open())
    {
        std::cout << "Where is File, Lebovski!?\n";
        std::exit(1);
    }
    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    int count = std::count(contents.begin(), contents.end(), ' ') + std::count(contents.begin(), contents.end(), '\n') + 1;
    std::cout << count / 4 << '\n';
}

void generateLattice(int linearSize) 
{}

void LatticeGPU::createDOS(int linearSize) 
{
    latticeConstructorDOSAdapter(G, E, M);
};

void LatticeGPU::calculate()
{
    if(E != nullptr && G != nullptr && M != nullptr)
    {
        calculateAdapter(G, E, M, x, y, mx, my);
    }
}

void Lattice::print()
{}
