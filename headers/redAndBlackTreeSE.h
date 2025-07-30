#ifndef RED_AND_BLACK_TREE_SE_H
#define RED_AND_BLACK_TREE_SE_H

#include <iostream>
#include <memory>
#include <tuple>

enum class ColorSE { RED, BLACK };

// Узел с составным ключом (энергия, спин) и частотой
struct NodeSE {
    float energy;
    float spin;
    unsigned long long degeneration;
    ColorSE color;
    std::shared_ptr<NodeSE> left, right, parent;

    NodeSE(float e, float m, int g) : 
        energy(e), spin(m), degeneration(g), 
        color(ColorSE::RED), 
        left(nullptr), right(nullptr), parent(nullptr) {}
};

class RBTreeSE 
{
private:
    float accuracy;
    std::shared_ptr<NodeSE> root;
    std::shared_ptr<NodeSE> nil;
    bool isLess(const std::pair<float, float>& key1, const std::pair<float, float>& key2) const;
    bool isEqual(const std::pair<float, float>& key1, const std::pair<float, float>& key2) const;
    void leftRotate(std::shared_ptr<NodeSE> x);
    void rightRotate(std::shared_ptr<NodeSE> y);
    void fixInsert(std::shared_ptr<NodeSE> k);
    void transplant(std::shared_ptr<NodeSE> u, std::shared_ptr<NodeSE> v);
    std::shared_ptr<NodeSE> minimum(std::shared_ptr<NodeSE> node);
    void fixDelete(std::shared_ptr<NodeSE> x);
    size_t sizeRecurCalc(std::shared_ptr<NodeSE> node);
    void inOrderToArrays(std::shared_ptr<NodeSE> node, unsigned long long *degeneracies, 
        float *energies, float *spins, size_t &idx);
public:
    RBTreeSE(float accuracy);
    void insert(unsigned long long  degeneration, float energy, float spin);
    void deleteNode(float energy, float spin);
    int search(float energy, float spin);
    size_t size();
    size_t toArrays(unsigned long long *degeneracies, float *energies, float *spins);
};

#endif
