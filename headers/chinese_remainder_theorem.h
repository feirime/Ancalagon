#ifndef STATSUM_EMC_SPACE_CHINESE_REMAINDER_THEOREM_H
#define STATSUM_EMC_SPACE_CHINESE_REMAINDER_THEOREM_H
#include <gmpxx.h>

typedef unsigned int uint16_cu;
typedef unsigned long long int uint64_cu;

extern bool spin_glass_read;

void modular_inverse_element_2_with_prime_modulo(mpz_t product, mpz_t x,  long int rec_number);
long int modular_gcd_v1(long int a, mpz_t const b);
void chinese_decryption(char *result, const unsigned long int *rem, unsigned int *prime_number, long int pr_n);

#endif
