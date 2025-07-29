#ifndef RED_AND_BLACK_TREE_H
#define RED_AND_BLACK_TREE_H

#include <memory>

// Цвет узла: RED или BLACK
enum class Color { RED, BLACK };

// Узел дерева, хранящий пару (энергия, частота)
struct Node {
    double energy;       // ключ - энергия
    int degeneracy;      // значение - вырождение
    Color color;        // цвет узла
    std::shared_ptr<Node> left, right, parent; // указатели
    
    // Конструктор
    Node(double e, int f) : energy(e), degeneracy(f), 
                           color(Color::RED), 
                           left(nullptr), right(nullptr), parent(nullptr) {}
};

class RBTree {
private:
    std::shared_ptr<Node> root;
    std::shared_ptr<Node> nil; // специальный листовой узел

    // Вспомогательные функции
    void leftRotate(std::shared_ptr<Node> x);
    void rightRotate(std::shared_ptr<Node> y);
    void fixInsert(std::shared_ptr<Node> k);
    void fixDelete(std::shared_ptr<Node> x);
    void transplant(std::shared_ptr<Node> u, std::shared_ptr<Node> v);
    std::shared_ptr<Node> minimum(std::shared_ptr<Node> node);
    int sizeRecurCalc(std::shared_ptr<Node> node);

public:
    RBTree() {
        nil = std::make_shared<Node>(0, 0);
        nil->color = Color::BLACK;
        root = nil;
    }
    
    // Вставка пары (энергия, частота)
    void insert(double energy, int degeneracy);
    
    // Удаление узла по энергии
    void deleteNode(double energy);
    
    // Поиск частоты по энергии
    int search(double energy);

    int RBTree::size();
};

#endif