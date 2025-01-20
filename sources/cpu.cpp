#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <vector>
#include "cpu.h"
#include "chinese_remainder_theorem.h"

void J_read(int n, int *J_h, int *J_v, const std::string& cell_name)
{
    float temp_J = 0;
    std::ifstream J_file(cell_name);
    if(!J_file.is_open())
        std::cout << "Where are integrals, Lebowski?\n";
    auto J_temp = new int [n * n * n * n];
    for(auto i = 0; i < n * n * n * n; i++)
    {
        J_file >> temp_J; 
        J_temp[i] = temp_J;
    }
    int k = 1;
    for(auto i = 0; i < n; ++i)
    {
        for(auto j = 0; j < (n - 1); j++)
        {
            J_h[i * (n - 1) + j] = J_temp[k];
            k += n * n + 1;
        }
        k += n * n + 1;
    }
    k = n;
    for(auto i = 0; i < n - 1; ++i)
    {
        for(auto j = 0; j < n; j++)
        {
            J_v[i * n + j] = J_temp[k];
            k += n * n + 1;
        }
    }
    delete [] J_temp;
}

void J_write(int n, const int *J_h, const int *J_v, int &sum_of_J, int sample_number)
{
    for(auto i = 0; i < n; ++i)
    {
        for (auto j = 0; j < (n - 1); ++j)
        {
            sum_of_J += J_h[i * (n - 1) + j];
        }
    }
    for(auto i = 0; i < (n - 1); ++i)
    {
        for (auto j = 0; j < n; ++j)
        {
            sum_of_J += J_v[i * n + j];
        }
    }
    std::ofstream cell_data("data/cell_write/cell" + std::to_string(n * n) + "_J" + std::to_string(sum_of_J) + "_"
    + std::to_string(sample_number) + ".dat");
    auto J_temp = new int [n * n * n * n];
    for(auto i = 0; i < n * n * n * n; ++i)
    {
        J_temp[i] = 0;
    }
    int k = 1;
    int l = n * n;
    for(auto i = 0; i < n; ++i)
    {
        for(auto j = 0; j < (n - 1); j++)
        {
            J_temp[k] = J_h[i * (n - 1) + j];
            J_temp[l] = J_h[i * (n - 1) + j];
            k += n * n + 1;
            l += n * n + 1;
        }
        k += n * n + 1;
        l += n * n + 1;
    }
    k = n;
    l = n * n * n;
    for(auto i = 0; i < n - 1; ++i)
    {
        for(auto j = 0; j < n; j++)
        {
            J_temp[k] = J_v[i * n + j];
            J_temp[l] = J_v[i * n + j];
            k += n * n + 1;
            l += n * n + 1;
        }
    }
    for(auto i = 0; i < n * n; ++i)
    {
        for(auto j = 0; j < n * n; ++j)
        {
            cell_data << J_temp[i * n * n + j] << " ";
        }
        cell_data << "\n";
    }
    delete [] J_temp;
}

void J_glass_generator(int n, int *J_h, int *J_v, int J_ferr)
{
    for(auto i = 0; i < n * (n - 1); ++i)
    {
        J_h[i] = -1;
        J_v[i] = -1;
    }
    int n_J = n * (n - 1);
    std::random_device dev;
    std::mt19937 rand_gen(dev());
    auto *ferr_idx_h = new int [n_J];
    auto *ferr_idx_v = new int [n_J];
    for(auto i = 0; i < n_J; i++)
    {
        ferr_idx_h[i] = i;
        ferr_idx_v[i] = i;
    }
    int rem_h = n_J;
    int rem_v = n_J;
    for(auto i = 0; i < J_ferr; i++)
    {
        std::uniform_int_distribution<std::mt19937::result_type> rand_n(0, 2 * n_J - 1 - i);
        unsigned int idx = rand_n(rand_gen);
        if(idx < rem_h)
        {
            J_h[ferr_idx_h[idx]] = 1;
            rem_h--;
            for(auto j = idx; j < rem_h; j++)
            {
                ferr_idx_h[j] = ferr_idx_h[j + 1];
            }
        }
        else
        {
            J_v[ferr_idx_v[idx - rem_h]] = 1;
            rem_v--;
            for(auto j = idx - rem_h; j < rem_v; j++)
            {
                ferr_idx_v[j] = ferr_idx_v[j + 1];
            }
        }
    }
    delete [] ferr_idx_h;
    delete [] ferr_idx_v;
}


void thread_init(int n, int8_cu *S)
{
    uint64_cu length = 1 << n;
    for (auto i = 0; i < length; i++)
    {
        int bit = i;
        for (auto j = 0; j < n; ++j)
        {
            S[i * n + j] = bit & 1 ? 1 : -1;
            bit >>= 1;
        }
    }
}

void out(int n, const uint64_cu *EMC, int E_max, int M_max, uint16_cu *prime_set,
         int J_antiferr, int sample_number, const std::string& name_add)
{
    int E_add = (E_max - 1) / 2;
    int M_add = (M_max - 1) / 2;
    std::string name;
    name = "data/dos_" + name_add;
    std::ofstream gem_out(name);
    gem_out << n << "\n";
    int size = 0;
    for (auto E = 0; E < E_max; ++E)
        for (auto M = 0; M < M_max; ++M)
            if (EMC[E * M_max * prime_n + M * prime_n] > 0)
            {
                size++;
            }
    gem_out << size << "\n";
    gem_out << J_antiferr << "\n";
    gem_out << sample_number << "\n";
    auto *rems = new unsigned long int [prime_n - 1];
    auto *G = new char [1];
    for (auto E = 0; E < E_max; ++E)
    {
        for (auto M = 0; M < M_max; ++M)
            if (EMC[E * M_max * prime_n + M * prime_n] > 0)
            {
                for (auto i = 1; i < prime_n; ++i)
                {
                    rems[i - 1] = EMC[E * M_max * prime_n + M * prime_n + i];
                }
                chinese_decryption(G, rems, prime_set, prime_n - 1);
                gem_out << G << " " << E - E_add << " " << M - M_add << "\n";
            }
    }
    gem_out.close();
    delete [] G;
    delete [] rems;
}
