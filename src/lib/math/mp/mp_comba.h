/*
* (C) 2024 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#ifndef BOTAN_MP_COMBA_H_
#define BOTAN_MP_COMBA_H_

namespace Botan {

template <size_t N, WordType W>
constexpr inline void comba_mul(W z[2*N], const W x[N], const W y[N]) {
   word w2 = 0, w1 = 0, w0 = 0;

   for(size_t i = 0; i != 2*N; ++i) {

      const size_t start = i + 1 < N ? 0 : i - N + 1;
      for(size_t j = start; j != std::min(N, i+1); ++j) {
         word3_muladd(&w2, &w1, &w0, x[j], y[i-j]);
      }
      z[i] = w0;
      w0 = w1;
      w1 = w2;
      w2 = 0;
   }

}

}

#endif
