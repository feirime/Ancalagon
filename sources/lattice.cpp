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

void Lattice::splitInit(float iteractionRadius, float splitSeed)
{
    this->iteractionRadius = iteractionRadius;
    latticeLinearSize = (int)sqrt((float)latticeSize);
    if(splitSeed == 1)
    {
        this->splitSeed = 1.0 / (float)latticeLinearSize;
    }
    else
        this->splitSeed = splitSeed;
    layer = 0;
    layers = (int)(1 / this->splitSeed);
    std::cout << "layers = " << layers << '\n';
}

void Lattice::layerPlusPlus()
{
    layer++;
}

void Lattice::mapMaker()
{}

//void Lattice::mapMaker() //теперь здесь насрано!
//{
//    float minX = *std::min_element(x, x + latticeSize);
//    float maxX = *std::max_element(x, x + latticeSize);
//    float minY = *std::min_element(y, y + latticeSize);
//    float maxY = *std::max_element(y, y + latticeSize);
//    float leftX = (maxX - minX) * splitSeed * layer;
//    float rightX = (maxX - minX) * splitSeed * (layer + 1);
//    std::cout << "leftXmain = " << leftX << " rightXmain = " << rightX << '\n';
//    layerMainSize = 0;
//    for(auto i = 0; i < latticeSize; i++)
//        if(leftX >= x[i] && x[i] < rightX) 
//            layerMainSize++;
//    latticeMainMalloc();
//    for(auto i = 0; i < latticeSize; i++)
//    {
//        size_t j = 0;
//        if(leftX >= x[i] && x[i] < rightX)
//        {
//            xMain[j] = x[i];
//            yMain[j] = y[i];
//            mxMain[j] = mx[i];
//            myMain[j] = my[i];
//            j++;
//        }
//    }
//    leftX = (maxX - minX) * splitSeed * (layer + 1);
//    rightX = (maxX - minX) * splitSeed * (layer + 2);
//    std::cout << "leftXadd = " << leftX << " rightXadd = " << rightX << '\n';
//    layerAddSize = 0;
//    for(auto i = 0; i < latticeSize; i++)
//        if(leftX >= x[i] && x[i] < rightX) 
//            layerAddSize++;
//    latticeAddMalloc();
//    for(auto i = 0; i < latticeSize; i++)
//    {
//        size_t j = 0;
//        if(leftX >= x[i] && x[i] < rightX)
//        {
//            xAdd[j] = x[i];
//            yAdd[j] = y[i];
//            mxAdd[j] = mx[i];
//            myAdd[j] = my[i];
//            j++;
//        }
//    }
//    layerResultSize = layerMainSize + layerAddSize;
//    std::cout << "layerMainSize = " << layerMainSize << '\n';
//    std::cout << "layerAddSize = " << layerAddSize << '\n';
//    std::cout << "layerResultSize = " << layerResultSize << '\n';
//    dosCopyMalloc();
//}

void Lattice::compress()
{
    size_t dosResultSize = pow(2, layerResultSize);
    for(auto i = 0; i < dosResultSize; i++)
    {
        for(auto j = 0; j < dosResultSize; j++)
        {}
    }
}

bool Lattice::isStart()
{
    if(layer == 0)
        return true;
    else
        return false;
}

bool Lattice::isEnd()
{
    if(layer == layers - 1)
        return true;
    else
        return false;
}

void Lattice::print()
{
    size_t dosResultSize = pow(2, layerResultSize);
    std::cout << "G " << "E " << "M " << '\n';
    for(int i = 0; i < dosResultSize; i++)
    {
        if(Gresult[i] == 0)
            continue;
        std::cout << Gresult[i] << " " << Eresult[i] << " " << Mresult[i] << '\n';
    }
}

Lattice::~Lattice() 
{
    std::cout << "Lattice destructor\n";
}
