/*
* (C) 2024 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#ifndef BOTAN_MP_MONTY_H_
#define BOTAN_MP_MONTY_H_

#include <botan/exceptn.h>
#include <botan/assert.h>
#include <botan/exceptn.h>
#include <botan/mem_ops.h>
#include <botan/internal/mp_core.h>
#include <botan/internal/ct_utils.h>

namespace Botan {

/**
* Compute the integer x such that (a*x) = -1 mod 2^N
*
* Throws an exception if input is even. If input is odd, then input
* and 2^N are relatively prime and the inverse exists.
*/
template <WordType W>
constexpr inline auto monty_inverse(W a) -> W {
   if(a % 2 == 0) {
      throw Invalid_Argument("monty_inverse only valid for odd integers");
   }

   /*
   * From "A New Algorithm for Inversion mod p^k" by Çetin Kaya Koç
   * https://eprint.iacr.org/2017/411.pdf sections 5 and 7.
   */

   W b = 1;
   W r = 0;

   for(size_t i = 0; i != WordInfo<W>::bits; ++i) {
      const W bi = b % 2;
      r >>= 1;
      r += bi << (WordInfo<W>::bits - 1);

      b -= a * bi;
      b >>= 1;
   }

   // Now invert in addition space
   r = (WordInfo<W>::max - r) + 1;

   return r;
}

/*
* Montgomery reduction
*
* Each of these functions makes the following assumptions:
*
* z_size == 2*p_size
* ws_size >= p_size + 1
*/
BOTAN_FUZZER_API void bigint_monty_redc_4(word z[8], const word p[4], word p_dash, word ws[]);
BOTAN_FUZZER_API void bigint_monty_redc_6(word z[12], const word p[6], word p_dash, word ws[]);
BOTAN_FUZZER_API void bigint_monty_redc_8(word z[16], const word p[8], word p_dash, word ws[]);
BOTAN_FUZZER_API void bigint_monty_redc_16(word z[32], const word p[16], word p_dash, word ws[]);
BOTAN_FUZZER_API void bigint_monty_redc_24(word z[48], const word p[24], word p_dash, word ws[]);
BOTAN_FUZZER_API void bigint_monty_redc_32(word z[64], const word p[32], word p_dash, word ws[]);

/*
* Montgomery reduction - product scanning form
*
* Algorithm 5 from "Energy-Efficient Software Implementation of Long
* Integer Modular Arithmetic"
* (https://www.iacr.org/archive/ches2005/006.pdf)
*
* See also
*
* https://eprint.iacr.org/2013/882.pdf
* https://www.microsoft.com/en-us/research/wp-content/uploads/1996/01/j37acmon.pdf
*/
template <WordType W>
constexpr inline void bigint_monty_redc_generic(W z[], size_t z_size,
                                                const W p[], size_t p_size,
                                                W p_dash, W ws[]) {

   BOTAN_ARG_CHECK(z_size >= 2 * p_size && p_size > 0, "Invalid sizes for bigint_monty_redc_generic");

   W w2 = 0, w1 = 0, w0 = 0;

   w0 = z[0];

   ws[0] = w0 * p_dash;

   word3_muladd(&w2, &w1, &w0, ws[0], p[0]);

   w0 = w1;
   w1 = w2;
   w2 = 0;

   for(size_t i = 1; i != p_size; ++i) {
      for(size_t j = 0; j < i; ++j) {
         word3_muladd(&w2, &w1, &w0, ws[j], p[i - j]);
      }

      word3_add(&w2, &w1, &w0, z[i]);

      ws[i] = w0 * p_dash;

      word3_muladd(&w2, &w1, &w0, ws[i], p[0]);

      w0 = w1;
      w1 = w2;
      w2 = 0;
   }

   for(size_t i = 0; i != p_size - 1; ++i) {
      for(size_t j = i + 1; j != p_size; ++j) {
         word3_muladd(&w2, &w1, &w0, ws[j], p[p_size + i - j]);
      }

      word3_add(&w2, &w1, &w0, z[p_size + i]);

      ws[i] = w0;

      w0 = w1;
      w1 = w2;
      w2 = 0;
   }

   word3_add(&w2, &w1, &w0, z[2 * p_size - 1]);

   ws[p_size - 1] = w0;
   ws[p_size] = w1;

   /*
   * The result might need to be reduced mod p. To avoid a timing
   * channel, always perform the subtraction. If in the compution
   * of x - p a borrow is required then x was already < p.
   *
   * x starts at ws[0] and is p_size+1 bytes long.
   * x - p starts at z[0] and is also p_size+1 bytes log
   *
   * If borrow was set then x was already < p and the subtraction
   * was not needed. In that case overwrite z[0:p_size] with the
   * original x in ws[0:p_size].
   *
   * We only copy out p_size in the final step because we know
   * the Montgomery result is < P
   */

   W borrow = bigint_sub3(z, ws, p_size + 1, p, p_size);

   BOTAN_DEBUG_ASSERT(borrow == 0 || borrow == 1);

   CT::conditional_assign_mem(borrow, z, ws, p_size);
   clear_mem(z + p_size, z_size - p_size);
}

/**
* Montgomery Reduction
* @param z integer to reduce, of size exactly 2*p_size. Output is in
* the first p_size+1 words, higher words are set to zero.
* @param p modulus
* @param p_size size of p
* @param p_dash Montgomery value
* @param ws array of at least p_size+1 words
* @param ws_size size of ws in words
*/
inline void bigint_monty_redc(word z[], const word p[], size_t p_size, word p_dash, word ws[], size_t ws_size) {
   const size_t z_size = 2 * p_size;

   BOTAN_ARG_CHECK(ws_size >= p_size + 1, "Montgomery workspace too small");

   return bigint_monty_redc_generic(z, z_size, p, p_size, p_dash, ws);
   /*
   if(p_size == 4) {
      bigint_monty_redc_4(z, p, p_dash, ws);
   } else if(p_size == 6) {
      bigint_monty_redc_6(z, p, p_dash, ws);
   } else if(p_size == 8) {
      bigint_monty_redc_8(z, p, p_dash, ws);
   } else if(p_size == 16) {
      bigint_monty_redc_16(z, p, p_dash, ws);
   } else if(p_size == 24) {
      bigint_monty_redc_24(z, p, p_dash, ws);
   } else if(p_size == 32) {
      bigint_monty_redc_32(z, p, p_dash, ws);
   } else {
      bigint_monty_redc_generic(z, z_size, p, p_size, p_dash, ws);
   }*/
}

}  // namespace Botan

#endif
