#ifndef RED_AND_BLACK_TREE_SE_H
#define RED_AND_BLACK_TREE_SE_H

#include <iostream>
#include <memory>
#include <tuple>

enum class ColorSE { RED, BLACK };

// Узел с составным ключом (энергия, спин) и частотой
struct NodeSE {
    double energy;
    int spin;
    int degradation;
    ColorSE color;
    std::shared_ptr<NodeSE> left, right, parent;

    NodeSE(double e, int m, int g) : 
        energy(e), spin(m), degradation(g), 
        color(ColorSE::RED), 
        left(nullptr), right(nullptr), parent(nullptr) {}
};

class RBTreeSE 
{
private:
    std::shared_ptr<NodeSE> root;
    std::shared_ptr<NodeSE> nil;

    // Сравнение двух составных ключей
    bool isLess(const std::pair<double, int>& key1, const std::pair<double, int>& key2) const;
    bool isEqual(const std::pair<double, int>& key1, const std::pair<double, int>& key2) const;
    void leftRotate(std::shared_ptr<NodeSE> x);
    void rightRotate(std::shared_ptr<NodeSE> y);
    void fixInsert(std::shared_ptr<NodeSE> k);
    void transplant(std::shared_ptr<NodeSE> u, std::shared_ptr<NodeSE> v);
    std::shared_ptr<NodeSE> minimum(std::shared_ptr<NodeSE> node);
    void fixDelete(std::shared_ptr<NodeSE> x);

public:
    RBTreeSE();

    // Вставка или обновление частоты
    void insert(double energy, int spin, int degradation = 1);
    // Удаление узла по ключу
    void deleteNode(double energy, int spin);
    // Поиск частоты по ключу
    int search(double energy, int spin);
};

#endif
