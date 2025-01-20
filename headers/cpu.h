#ifndef cpu_h
#define cpu_h

typedef unsigned long long int uint64_cu;
typedef unsigned int uint16_cu;
typedef signed char int8_cu;

extern bool spin_glass_read;
extern int const prime_n;

void J_read(int n, int*, int*, const std::string&);
void J_write(int n, const int *J_h, const int *J_v, int &sum_of_J, int sample_number);
void J_glass_generator(int n, int *J_h, int *J_v, int J_ferr);
void thread_init(int n, int8_cu*);
void out(int n, const uint64_cu *EMC, int E_max, int M_max, uint16_cu *prime_set,
         int J_antiferr, int sample_number, const std::string& name_add);

#endif
