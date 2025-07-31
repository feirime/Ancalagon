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
        std::string temp;
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

void Lattice::init(float iteractionRadius, float accuracy, float splitSeed)
{
    this->iteractionRadius = iteractionRadius;
    this->accuracy = accuracy;
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

void Lattice::mapMaker() //теперь здесь насрано!
{
    float minX = *std::min_element(x, x + latticeSize);
    float maxX = *std::max_element(x, x + latticeSize);
    float minY = *std::min_element(y, y + latticeSize);
    float maxY = *std::max_element(y, y + latticeSize);
    float leftX = (maxX - minX) * splitSeed * layer;
    float rightX = (maxX - minX) * splitSeed * (layer + 1);
    std::cout << "leftXmain = " << leftX << " rightXmain = " << rightX << '\n';
    layerMainSize = 0;
    for(auto i = 0; i < latticeSize; i++)
        if(leftX >= x[i] && x[i] < rightX) 
            layerMainSize++;
    latticeMainMalloc();
    for(auto i = 0; i < latticeSize; i++)
    {
        size_t j = 0;
        if(leftX >= x[i] && x[i] < rightX)
        {
            xMain[j] = x[i];
            yMain[j] = y[i];
            mxMain[j] = mx[i];
            myMain[j] = my[i];
            j++;
        }
    }
    leftX = (maxX - minX) * splitSeed * (layer + 1);
    rightX = (maxX - minX) * splitSeed * (layer + 2);
    std::cout << "leftXadd = " << leftX << " rightXadd = " << rightX << '\n';
    layerAddSize = 0;
    for(auto i = 0; i < latticeSize; i++)
        if(leftX >= x[i] && x[i] < rightX) 
            layerAddSize++;
    latticeAddMalloc();
    for(auto i = 0; i < latticeSize; i++)
    {
        size_t j = 0;
        if(leftX >= x[i] && x[i] < rightX)
        {
            xAdd[j] = x[i];
            yAdd[j] = y[i];
            mxAdd[j] = mx[i];
            myAdd[j] = my[i];
            j++;
        }
    }
    layerResultSize = layerMainSize + layerAddSize;
    std::cout << "layerMainSize = " << layerMainSize << '\n';
    std::cout << "layerAddSize = " << layerAddSize << '\n';
    std::cout << "layerResultSize = " << layerResultSize << '\n';
    dosMalloc();
}

void Lattice::compress()
{
    size_t dosResultSize = pow(2, layerResultSize);
    for(auto i = 0; i < dosResultSize; i++)
    {
        for(auto j = i + 1; j < dosResultSize; j++)
        {
            if(Gresult[i] != 0 && Gresult[j] != 0 && abs(Eresult[i] - Eresult[j]) < 1e-6 && abs(Mresult[i] - Mresult[j]) < 1e-6)
            {
                Gresult[i] += Gresult[j];
                Gresult[j] = 0;
            }
        }
    }
}

void Lattice::compressRBTree()
{
    RBTree tree;
    dosResultSize = pow(2, layerResultSize);
    for(auto i = 0; i < dosResultSize; i++)
    {
        if(Gresult[i] != 0)
            tree.insert(Gresult[i], Eresult[i]);
    }
    dosDeleteResult();
    dosResultSize = tree.size();
    std::cout << "dosResultSize = " << dosResultSize << '\n';
    dosMallocResult(dosResultSize);
    tree.toArrays(Gresult, Eresult);
}

void Lattice::compressRBTreeSE()
{
    RBTreeSE tree(accuracy);
    dosResultSize = pow(2, layerResultSize);
    for(auto i = 0; i < dosResultSize; i++)
    {
        if(Gresult[i] != 0)
            tree.insert(Gresult[i], Eresult[i], Mresult[i]);
    }
    dosDeleteResult();
    dosResultSize = tree.size();
    std::cout << "dosResultSize = " << dosResultSize << '\n';
    dosMallocResult(dosResultSize);
    tree.toArrays(Gresult, Eresult, Mresult);
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

void Lattice::brutforce()
{
    printf("Iteraction radius = %f\n", iteractionRadius);
    float cosin45 = sqrt(2) / 2;
    for(auto conf = 0; conf < dosResultSize; conf++)
    {
        Gresult[conf] = 0;
        Eresult[conf] = 0;
        Mresult[conf] = 0;
        for(auto i = 0; i < latticeSize; i++)
        {
            float mxi = mx[i];
            float myi = my[i];
            if(conf >> i & 1)
            {
                mxi *= -1;
                myi *= -1;
            }
            for(auto j = i + 1; j < latticeSize; j++)
            {
                float xij = x[i] - x[j];
                float yij = y[i] - y[j];
                float r = sqrt(xij * xij + yij * yij);
                if(r > iteractionRadius)
                    continue;
                Gresult[conf] = 1;
                float mxj = mx[j];
                float myj = my[j];
                if(conf >> j & 1)
                {
                    mxj *= -1;
                    myj *= -1;
                }
                //float distanse = sqrt(pow(x[i] - x[j], 2) + pow(y[i] - y[j], 2));
                Eresult[conf] += (mxi * mxj + myi * myj) / pow(r, 3) - 3 * (mxi * xij + myi * yij) * (mxj * xij + myj * yij) / pow(r, 5);
            }
            Mresult[conf] += mxi * cosin45 + myi * cosin45; //проекция на диагональ XY
        }
    }
}

void Lattice::print()
{
    //std::cout << "G " << "E " << "M " << '\n';
    //for(int i = 0; i < dosResultSize; i++)
    //{
    //    if(Gresult[i] == 0)
    //        continue;
    //    std::cout << Gresult[i] << " " << Eresult[i] << " " << Mresult[i] << '\n';
    //}
    float eGs = 1e7;
    for(auto i = 0; i < dosResultSize; i++)
    {
        if(Eresult[i] < eGs & Gresult[i] != 0)
        {
            eGs = Eresult[i];
        }
    }
    printf("Egs = %f\n", eGs);
    for(auto i = 0; i < dosResultSize; i++)
    {
        if(Gresult[i] != 0 & abs(Eresult[i] - eGs) < 1e-6)
        {
            std::cout << "G = " << Gresult[i] << ", E = " << Eresult[i] << ", M = " << Mresult[i] << "\n";
        }
    }
}

Lattice::~Lattice() 
{
    std::cout << "Lattice destructor\n";
}