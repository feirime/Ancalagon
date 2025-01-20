#include "chinese_remainder_theorem.h"

void modular_inverse_element_2_with_prime_modulo(mpz_t product, mpz_t x,  long int rec_number)
{
    mpz_set_ui(product, 1);
    for(long long int counter = 0;  counter < rec_number - 2;  counter++)
    {
        mpz_mul(product, product, x);
        if(mpz_cmp_ui(product, rec_number)>=0)
        {
            mpz_fdiv_r_ui(product, product, rec_number);
        }
    }
}

long int modular_gcd_v1(long int a, mpz_t const b)
{
    long int result;
    mpz_t t;
    mpz_init_set_ui(t, 0);
    mpz_t mpa;
    mpz_init_set_ui(mpa, a);
    mpz_t mpb;
    mpz_init_set(mpb, b);
    while(-mpz_cmp_ui(mpb, 0))
    {
        mpz_set(t, mpb);
        mpz_cdiv_r(mpb, mpa, mpb);
        mpz_set(mpa, t);
    }
    result = mpz_get_ui(mpa);
    mpz_clear(t);
    mpz_clear(mpa);
    mpz_clear(mpb);
    return result;
}

void chinese_decryption(char *result, const unsigned long int *rem, unsigned int *prime_number, long int pr_n)
{
    mpz_t total_mult;
    mpz_init_set_ui(total_mult, prime_number[0]);
    for (long long int i = 1; i < pr_n; i++)
    {
        mpz_mul_ui(total_mult, total_mult, prime_number[i]);
    }
    mpz_t Mi;
    mpz_init_set_ui(Mi, 0);
    long int my_gcd;
    long int num_rec;
    unsigned long int rem_rec;
    mpz_t mult_rec;
    mpz_init(mult_rec);
    mpz_t M_rec;
    mpz_init_set_ui(M_rec, 0);
    mpz_t answer;
    mpz_init_set_ui(answer, 0);
    for(long long int i=0; i < pr_n; i++)
    {
        mpz_fdiv_q_ui(Mi, total_mult, prime_number[i]);
        my_gcd = modular_gcd_v1(prime_number[i], Mi);
        num_rec = prime_number[i] / my_gcd;
        rem_rec = rem[i] / my_gcd;
        mpz_fdiv_q_ui(mult_rec, total_mult, num_rec);
        modular_inverse_element_2_with_prime_modulo(M_rec, mult_rec, num_rec);
        mpz_mul_ui(M_rec, M_rec, rem_rec);
        mpz_fdiv_r_ui(M_rec, M_rec, num_rec);
        mpz_mul(M_rec, M_rec, Mi);
        mpz_add(answer, answer, M_rec);
    }
    mpz_fdiv_r(answer, answer, total_mult);
    mpz_get_str(result, 10, answer);
    mpz_clear(total_mult);
    mpz_clear(Mi);
    mpz_clear(mult_rec);
    mpz_clear(M_rec);
    mpz_clear(answer);
}
