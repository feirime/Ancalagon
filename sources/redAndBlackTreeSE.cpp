#include "redAndBlackTreeSE.h"

bool RBTreeSE::isLess(const std::pair<float, float>& key1, const std::pair<float, float>& key2) const 
    {
        if (key1.first != key2.first)
            return key1.first < key2.first;
        return key1.second < key2.second;
    }

bool RBTreeSE::isEqual(const std::pair<float, float>& key1, const std::pair<float, float>& key2) const 
{
    return abs(key1.first - key2.first) < accuracy && abs(key1.second - key2.second) < accuracy;
}

void RBTreeSE::leftRotate(std::shared_ptr<NodeSE> x) 
{
    std::shared_ptr<NodeSE> y = x->right;
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

void RBTreeSE::rightRotate(std::shared_ptr<NodeSE> y) 
{
    std::shared_ptr<NodeSE> x = y->left;
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


void RBTreeSE::fixInsert(std::shared_ptr<NodeSE> k) 
{
    std::shared_ptr<NodeSE> u;
    while (k->parent->color == ColorSE::RED) 
    {
        if (k->parent == k->parent->parent->right) 
        {
            u = k->parent->parent->left;
            if (u->color == ColorSE::RED) 
            {
                u->color = ColorSE::BLACK;
                k->parent->color = ColorSE::BLACK;
                k->parent->parent->color = ColorSE::RED;
                k = k->parent->parent;
            } else 
            {
                if (k == k->parent->left) 
                {
                    k = k->parent;
                    rightRotate(k);
                }
                k->parent->color = ColorSE::BLACK;
                k->parent->parent->color = ColorSE::RED;
                leftRotate(k->parent->parent);
            }
        } else 
        {
            u = k->parent->parent->right;
            if (u->color == ColorSE::RED) 
            {
                u->color = ColorSE::BLACK;
                k->parent->color = ColorSE::BLACK;
                k->parent->parent->color = ColorSE::RED;
                k = k->parent->parent;
            } 
            else 
            {
                if (k == k->parent->right) 
                {
                    k = k->parent;
                    leftRotate(k);
                }
                k->parent->color = ColorSE::BLACK;
                k->parent->parent->color = ColorSE::RED;
                rightRotate(k->parent->parent);
            }
        }
        if (k == root) break;
    }
    root->color = ColorSE::BLACK;
}

void RBTreeSE::transplant(std::shared_ptr<NodeSE> u, std::shared_ptr<NodeSE> v) 
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

std::shared_ptr<NodeSE> RBTreeSE::minimum(std::shared_ptr<NodeSE> node) 
{
    while (node->left != nil) 
    {
        node = node->left;
    }
    return node;
}

void RBTreeSE::fixDelete(std::shared_ptr<NodeSE> x) 
{
    std::shared_ptr<NodeSE> s;
    while (x != root && x->color == ColorSE::BLACK) 
    {
        if (x == x->parent->left) 
        {
            s = x->parent->right;
            if (s->color == ColorSE::RED) 
            {
                s->color = ColorSE::BLACK;
                x->parent->color = ColorSE::RED;
                leftRotate(x->parent);
                s = x->parent->right;
            }
            
            if (s->left->color == ColorSE::BLACK && s->right->color == ColorSE::BLACK) 
            {
                s->color = ColorSE::RED;
                x = x->parent;
            } 
            else 
            {
                if (s->right->color == ColorSE::BLACK) 
                {
                    s->left->color = ColorSE::BLACK;
                    s->color = ColorSE::RED;
                    rightRotate(s);
                    s = x->parent->right;
                }
                s->color = x->parent->color;
                x->parent->color = ColorSE::BLACK;
                s->right->color = ColorSE::BLACK;
                leftRotate(x->parent);
                x = root;
            }
        } 
        else 
        {
            s = x->parent->left;
            if (s->color == ColorSE::RED) 
            {
                s->color = ColorSE::BLACK;
                x->parent->color = ColorSE::RED;
                rightRotate(x->parent);
                s = x->parent->left;
            }
            
            if (s->right->color == ColorSE::BLACK && s->left->color == ColorSE::BLACK) 
            {
                s->color = ColorSE::RED;
                x = x->parent;
            } 
            else 
            {
                if (s->left->color == ColorSE::BLACK) 
                {
                    s->right->color = ColorSE::BLACK;
                    s->color = ColorSE::RED;
                    leftRotate(s);
                    s = x->parent->left;
                }
                s->color = x->parent->color;
                x->parent->color = ColorSE::BLACK;
                s->left->color = ColorSE::BLACK;
                rightRotate(x->parent);
                x = root;
            }
        }
    }
    x->color = ColorSE::BLACK;
}

size_t RBTreeSE::sizeRecurCalc(std::shared_ptr<NodeSE> node) 
{
    if (node == nil) 
    {
        return 0;
    }
    return 1 + sizeRecurCalc(node->left) + sizeRecurCalc(node->right);
}

void RBTreeSE::inOrderToArrays(std::shared_ptr<NodeSE> node, unsigned long long *degeneracies, 
    float *energies, float *spins, size_t &idx) 
{
    if (node == nil) 
        return;
    inOrderToArrays(node->left, degeneracies, energies, spins, idx);
    energies[idx] = node->energy;
    degeneracies[idx] = node->degeneration;
    idx++;
    inOrderToArrays(node->right, degeneracies, energies, spins, idx);
}

//---------------------------------------------Public---------------------------------------------

RBTreeSE::RBTreeSE(float accuracy) 
{
    nil = std::make_shared<NodeSE>(0, 0, 0);
    nil->color = ColorSE::BLACK;
    root = nil;
}

void RBTreeSE::insert(unsigned long long degeneration, float energy, float spin) 
{
    auto key = std::make_pair(energy, spin);
    std::shared_ptr<NodeSE> node = std::make_shared<NodeSE>(energy, spin, degeneration);
    node->parent = nil;
    node->left = nil;
    node->right = nil;
    
    std::shared_ptr<NodeSE> y = nil;
    std::shared_ptr<NodeSE> x = root;
    
    while (x != nil) 
    {
        y = x;
        auto currentKey = std::make_pair(x->energy, x->spin);
        
        if (isLess(key, currentKey)) 
        {
            x = x->left;
        } 
        else if (isEqual(key, currentKey)) 
        {
            // Если ключ уже существует, увеличиваем частоту
            x->degeneration += degeneration;
            return;
        } 
        else 
        {
            x = x->right;
        }
    }
    
    node->parent = y;
    if (y == nil) 
    {
        root = node;
    } 
    else if (isLess(key, std::make_pair(y->energy, y->spin))) 
    {
        y->left = node;
    } 
    else 
    {
        y->right = node;
    }
    
    if (node->parent == nil) 
    {
        node->color = ColorSE::BLACK;
        return;
    }
    
    if (node->parent->parent == nil) 
    {
        return;
    }
    fixInsert(node);
}

void RBTreeSE::deleteNode(float energy, float spin) 
{
    auto key = std::make_pair(energy, spin);
    std::shared_ptr<NodeSE> z = root;
    std::shared_ptr<NodeSE> x, y;
    
    // Поиск узла для удаления
    while (z != nil) 
    {
        auto currentKey = std::make_pair(z->energy, z->spin);
        if (isEqual(key, currentKey)) 
        {
            break;
        }
        z = isLess(key, currentKey) ? z->left : z->right;
    }

    if (z == nil) 
    {
        std::cout << "Ключ не найден в дереве" << '\n';
        return;
    }
    
    y = z;
    ColorSE y_original_color = y->color;
    
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
    
    if (y_original_color == ColorSE::BLACK) 
    {
        fixDelete(x);
    }
}

int RBTreeSE::search(float energy, float spin) 
{
    auto key = std::make_pair(energy, spin);
    std::shared_ptr<NodeSE> current = root;
    
    while (current != nil) 
    {
        auto currentKey = std::make_pair(current->energy, current->spin);
        
        if (isEqual(key, currentKey)) 
        {
            return current->degeneration;
        } 
        else if (isLess(key, currentKey)) 
        {
            current = current->left;
        } 
        else 
        {
            current = current->right;
        }
    }
    return 0; // не найдено
}

size_t RBTreeSE::size()
{
    return sizeRecurCalc(root);
}

size_t RBTreeSE::toArrays(unsigned long long *degeneracies, float *energies, float *spins) 
{
    size_t idx = 0;
    inOrderToArrays(root, degeneracies, energies, spins, idx);
    return idx;
}
