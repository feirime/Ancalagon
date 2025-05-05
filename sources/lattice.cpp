#include "lattice.h"
#include <algorithm>

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

int Lattice::read(std::string fileName) 
{
    std::ifstream fileContents(fileName);
    if(!fileContents.is_open())
    {
        std::cout << "Where is File, Lebovski!?\n";
        std::exit(1);
    }
    std::string contents((std::istreambuf_iterator<char>(fileContents)), std::istreambuf_iterator<char>());
    int count = std::count(contents.begin(), contents.end(), ' ') + std::count(contents.begin(), contents.end(), '\t') + std::count(contents.begin(), contents.end(), '\n');
    if(count == 0)
    {
        std::cout << "File is empty\n";
        std::exit(1);
    }
    latticeSize = (count - 4) / 4;
    std::ifstream file(fileName);
    printf("count = %d, latticeSize = %d\n", count, latticeSize);
    fileContents.close();
    latticeMalloc();
    double temp = 0;
    for(auto i = 0; i < 4; i++)
    {
        float temp = 0;
        file >> temp;
    }
    for(auto i = 0; i < latticeSize; i++)
    {
        file >> x[i];
        file >> y[i];
        file >> mx[i];
        file >> my[i];
    }
    file.close();
    return latticeSize;
}

void Lattice::initializeLattice(float iteractionRadius, float splitSeed)
{
    this->iteractionRadius = iteractionRadius;
    this->splitSeed = splitSeed;
    latticeLinearSize = (int)sqrt((float)latticeSize);
    layer = 0;
    layers = latticeLinearSize / splitSeed;
}

void Lattice::addConfigure()
{
    layer++;
    mapMaker();
}

void Lattice::compress()
{}

void Lattice::mapMaker()
{
    float minX = *std::min_element(x, x + latticeSize);
    float maxX = *std::max_element(x, x + latticeSize);
    float minY = *std::min_element(y, y + latticeSize);
    float maxY = *std::max_element(y, y + latticeSize);
    layerMainSize = 0;
    for(auto i = 0; i < latticeSize; i++)
        if(x[i] > layer * splitSeed && x[i] < (layer + 1) * splitSeed) 
            layerMainSize++;
    latticeMainMalloc();
    for(auto i = 0; i < latticeSize; i++)
    {
        size_t j = 0;
        if(x[i] > layer * splitSeed && x[i] < (layer + 1) * splitSeed)
        {
            xMain[j] = x[i];
            yMain[j] = y[i];
            mxMain[j] = mx[i];
            myMain[j] = my[i];
            j++;
        }
    }
    layerAddSize = 0;
    for(auto i = 0; i < latticeSize; i++)
        if(x[i] > (layer + 1) * splitSeed && x[i] < (layer + 2) * splitSeed) 
            layerAddSize++;
    latticeAddMalloc();
    layerResultSize = layerMainSize + layerAddSize;
}

bool Lattice::isEnd()
{
    if(layer == layers)
        return true;
    else
        return false;
}

void Lattice::print()
{
    std::cout << "x: " << "y " << "mx " << "my " << '\n';
    for(int i = 0; i < latticeSize; i++)
    {
        std::cout << x[i] << " " << y[i] << " " << mx[i] << " " << my[i] << '\n';
    }
}

Lattice::~Lattice() 
{
    std::cout << "Lattice destructor\n";
}
