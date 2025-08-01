#ifndef RED_AND_BLACK_TREE_H
#define RED_AND_BLACK_TREE_H

#include <memory>

// Цвет узла: RED или BLACK
enum class Color { RED, BLACK };

// Узел дерева, хранящий пару (энергия, частота)
struct Node {
    float energy;       // ключ - энергия
    unsigned long long int degeneracy;      // значение - вырождение
    Color color;        // цвет узла
    std::shared_ptr<Node> left, right, parent;
    Node(float e, unsigned long long int f) : energy(e), degeneracy(f), 
                           color(Color::RED), 
                           left(nullptr), right(nullptr), parent(nullptr) {}
};

class RBTree {
private:
    std::shared_ptr<Node> root;
    std::shared_ptr<Node> nil;
    void leftRotate(std::shared_ptr<Node> x);
    void rightRotate(std::shared_ptr<Node> y);
    void fixInsert(std::shared_ptr<Node> k);
    void fixDelete(std::shared_ptr<Node> x);
    void transplant(std::shared_ptr<Node> u, std::shared_ptr<Node> v);
    std::shared_ptr<Node> minimum(std::shared_ptr<Node> node);
    size_t sizeRecurCalc(std::shared_ptr<Node> node);
    void inOrderToArrays(std::shared_ptr<Node> node, 
        unsigned long long int *degeneracies, float *energies, size_t &idx);
public:
    RBTree();
    void insert(unsigned long long int degeneracy, float energy);
    void deleteNode(float energy);
    int search(float energy);
    size_t size();
    size_t toArrays(unsigned long long int *degeneracies, float *energies);
};

#endif