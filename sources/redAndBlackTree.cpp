#include "redAndBlackTree.h"
#include <iostream>

// Левый поворот
void RBTree::leftRotate(std::shared_ptr<Node> x) 
{
    std::shared_ptr<Node> y = x->right;
    x->right = y->left;
    
    if (y->left != nil) 
    {
        y->left->parent = x;
    }
    
    y->parent = x->parent;
    
    if (x->parent == nil) 
    {
        root = y;
    } 
    else if (x == x->parent->left) 
    {
        x->parent->left = y;
    } 
    else 
    {
        x->parent->right = y;
    }
    
    y->left = x;
    x->parent = y;
}

// Правый поворот
void RBTree::rightRotate(std::shared_ptr<Node> y) 
{
    std::shared_ptr<Node> x = y->left;
    y->left = x->right;
    
    if (x->right != nil) 
    {
        x->right->parent = y;
    }
    
    x->parent = y->parent;
    
    if (y->parent == nil) 
    {
        root = x;
    } 
    else if (y == y->parent->right) 
    {
        y->parent->right = x;
    } 
    else 
    {
        y->parent->left = x;
    }
    
    x->right = y;
    y->parent = x;
}

// Балансировка после вставки
void RBTree::fixInsert(std::shared_ptr<Node> k) 
{
    std::shared_ptr<Node> u;
    while (k->parent->color == Color::RED) 
    {
        if (k->parent == k->parent->parent->right) 
        {
            u = k->parent->parent->left;
            if (u->color == Color::RED) 
            {
                u->color = Color::BLACK;
                k->parent->color = Color::BLACK;
                k->parent->parent->color = Color::RED;
                k = k->parent->parent;
            } 
            else 
            {
                if (k == k->parent->left) 
                {
                    k = k->parent;
                    rightRotate(k);
                }
                k->parent->color = Color::BLACK;
                k->parent->parent->color = Color::RED;
                leftRotate(k->parent->parent);
            }
        } 
        else 
        {
            u = k->parent->parent->right;
            if (u->color == Color::RED) 
            {
                u->color = Color::BLACK;
                k->parent->color = Color::BLACK;
                k->parent->parent->color = Color::RED;
                k = k->parent->parent;
            } else 
            {
                if (k == k->parent->right) 
                {
                    k = k->parent;
                    leftRotate(k);
                }
                k->parent->color = Color::BLACK;
                k->parent->parent->color = Color::RED;
                rightRotate(k->parent->parent);
            }
        }
        if (k == root) break;
    }
    root->color = Color::BLACK;
}

RBTree::RBTree() 
{
    nil = std::make_shared<Node>(0, 0);
    nil->color = Color::BLACK;
    root = nil;
}
// Вставка нового узла
void RBTree::insert(unsigned long long int degeneracy, float energy) 
{
    std::shared_ptr<Node> node = std::make_shared<Node>(energy, degeneracy);
    node->parent = nil;
    node->left = nil;
    node->right = nil;
    
    std::shared_ptr<Node> y = nil;
    std::shared_ptr<Node> x = root;
    
    while (x != nil) 
    {
        y = x;
        if (node->energy < x->energy) 
        {
            x = x->left;
        } 
        else if (node->energy > x->energy) 
        {
            x = x->right;
        } 
        else 
        {
            // Если энергия уже существует, увеличиваем частоту
            x->degeneracy += degeneracy;
            return;
        }
    }
    
    node->parent = y;
    if (y == nil) 
    {
        root = node;
    } 
    else if (node->energy < y->energy) 
    {
        y->left = node;
    } 
    else 
    {
        y->right = node;
    }
    
    if (node->parent == nil) 
    {
        node->color = Color::BLACK;
        return;
    }
    
    if (node->parent->parent == nil) 
    {
        return;
    }
    
    fixInsert(node);
}

// Поиск частоты по энергии
int RBTree::search(float energy) 
{
    std::shared_ptr<Node> current = root;
    while (current != nil) 
    {
        if (energy < current->energy) 
        {
            current = current->left;
        } 
        else if (energy > current->energy) 
        {
            current = current->right;
        } 
        else 
        {
            return current->degeneracy;
        }
    }
    return 0; // не найдено
}

// Вспомогательные функции для удаления
void RBTree::transplant(std::shared_ptr<Node> u, std::shared_ptr<Node> v) 
{
    if (u->parent == nil) 
    {
        root = v;
    } 
    else if (u == u->parent->left) 
    {
        u->parent->left = v;
    } 
    else 
    {
        u->parent->right = v;
    }
    v->parent = u->parent;
}

std::shared_ptr<Node> RBTree::minimum(std::shared_ptr<Node> node) 
{
    while (node->left != nil) 
    {
        node = node->left;
    }
    return node;
}

void RBTree::fixDelete(std::shared_ptr<Node> x) 
{
    std::shared_ptr<Node> s;
    while (x != root && x->color == Color::BLACK) 
    {
        if (x == x->parent->left) 
        {
            s = x->parent->right;
            if (s->color == Color::RED) 
            {
                s->color = Color::BLACK;
                x->parent->color = Color::RED;
                leftRotate(x->parent);
                s = x->parent->right;
            }
            
            if (s->left->color == Color::BLACK && s->right->color == Color::BLACK) 
            {
                s->color = Color::RED;
                x = x->parent;
            } 
            else 
            {
                if (s->right->color == Color::BLACK) 
                {
                    s->left->color = Color::BLACK;
                    s->color = Color::RED;
                    rightRotate(s);
                    s = x->parent->right;
                }
                s->color = x->parent->color;
                x->parent->color = Color::BLACK;
                s->right->color = Color::BLACK;
                leftRotate(x->parent);
                x = root;
            }
        } 
        else 
        {
            s = x->parent->left;
            if (s->color == Color::RED) 
            {
                s->color = Color::BLACK;
                x->parent->color = Color::RED;
                rightRotate(x->parent);
                s = x->parent->left;
            }
            
            if (s->right->color == Color::BLACK && s->left->color == Color::BLACK) 
            {
                s->color = Color::RED;
                x = x->parent;
            } 
            else 
            {
                if (s->left->color == Color::BLACK) 
                {
                    s->right->color = Color::BLACK;
                    s->color = Color::RED;
                    leftRotate(s);
                    s = x->parent->left;
                }
                s->color = x->parent->color;
                x->parent->color = Color::BLACK;
                s->left->color = Color::BLACK;
                rightRotate(x->parent);
                x = root;
            }
        }
    }
    x->color = Color::BLACK;
}

// Удаление узла по энергии
void RBTree::deleteNode(float energy) 
{
    std::shared_ptr<Node> z = root;
    std::shared_ptr<Node> x, y;
    
    // Поиск узла для удаления
    while (z != nil) 
    {
        if (z->energy == energy) 
        {
            break;
        }
        z = (energy < z->energy) ? z->left : z->right;
    }
    
    if (z == nil) 
    {
        std::cout << "Энергия не найдена в дереве" << '\n';
        return;
    }
    
    y = z;
    Color y_original_color = y->color;
    
    if (z->left == nil) 
    {
        x = z->right;
        transplant(z, z->right);
    } 
    else if (z->right == nil) 
    {
        x = z->left;
        transplant(z, z->left);
    } 
    else 
    {
        y = minimum(z->right);
        y_original_color = y->color;
        x = y->right;
        
        if (y->parent == z) 
        {
            x->parent = y;
        } 
        else 
        {
            transplant(y, y->right);
            y->right = z->right;
            y->right->parent = y;
        }
        
        transplant(z, y);
        y->left = z->left;
        y->left->parent = y;
        y->color = z->color;
    }
    
    if (y_original_color == Color::BLACK) 
    {
        fixDelete(x);
    }
}

size_t RBTree::size()
{
    return sizeRecurCalc(root);
}

size_t RBTree::sizeRecurCalc(std::shared_ptr<Node> node) 
{
    if (node == nil) 
    {
        return 0;
    }
    return 1 + sizeRecurCalc(node->left) + sizeRecurCalc(node->right);
}

size_t RBTree::toArrays(unsigned long long int *degeneracies, float *energies) 
{
    size_t idx = 0;
    inOrderToArrays(root, degeneracies, energies, idx);
    return idx;
}

void RBTree::inOrderToArrays(std::shared_ptr<Node> node, 
    unsigned long long int *degeneracies, float *energies, size_t &idx) 
{
    if (node == nil) 
        return;
    inOrderToArrays(node->left, degeneracies, energies, idx);
    energies[idx] = node->energy;
    degeneracies[idx] = node->degeneracy;
    idx++;
    inOrderToArrays(node->right, degeneracies, energies, idx);
}
