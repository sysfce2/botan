/*
* (C) 2026 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <botan/internal/ctr.h>

#include <botan/assert.h>
#include <botan/internal/isa_extn.h>
#include <immintrin.h>

namespace Botan {

/*
* CTR mode counter increment and output XOR for bulk operations,
* specialized for block size of 16 and counter width of 32 bits.
*/
BOTAN_FN_ISA_AVX2
size_t CTR_BE::ctr_proc_bs16_ctr4_avx2(const uint8_t* in, uint8_t* out, size_t length) {
   BOTAN_ASSERT_NOMSG(m_pad.size() % 64 == 0);
   BOTAN_DEBUG_ASSERT(m_counter.size() == m_pad.size());

   const size_t pad_size = m_pad.size();
   if(length < pad_size) {
      return 0;
   }

   const size_t ctr_blocks = m_ctr_blocks;

   // NOLINTBEGIN(portability-simd-intrinsics)

   /*
   * Byte swap table that just swaps the counter bytes and not the nonce bytes
   */
   // clang-format off
   const __m256i bswap_ctr = _mm256_set_epi8(
      12, 13, 14, 15, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
      12, 13, 14, 15, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
   );
   // clang-format on

   // Load the starting counter value, bswap the counter field itself so we can add
   const __m256i starting_ctr = _mm256_shuffle_epi8(
      _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(m_counter.data()))), bswap_ctr);

   // Counter is incremented 2 blocks at a time
   const __m256i inc2 = _mm256_set_epi32(2, 0, 0, 0, 2, 0, 0, 0);

   const uint32_t N = static_cast<uint32_t>(ctr_blocks);
   __m256i batch_ctr = _mm256_add_epi32(starting_ctr, _mm256_set_epi32(N + 1, 0, 0, 0, N, 0, 0, 0));
   const uint8_t* pad_buf = m_pad.data();
   uint8_t* ctr_buf = m_counter.data();

   const size_t pad_block_pairs = pad_size / 32;    // denominated in bytes
   const size_t ctr_blocks_pairs = ctr_blocks / 2;  // denominated in 16 byte blocks

   size_t processed = 0;

   while(length >= pad_size) {
      // XOR m_pad into the input
      for(size_t i = 0; i != pad_block_pairs; ++i) {
         _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 32 * i),
                             _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pad_buf + 32 * i)),
                                              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + 32 * i))));
      }

      in += pad_size;
      out += pad_size;
      length -= pad_size;
      processed += pad_size;

      // Update the counter buffer
      for(size_t i = 0; i != ctr_blocks_pairs; ++i) {
         _mm256_storeu_si256(reinterpret_cast<__m256i*>(ctr_buf + i * 32), _mm256_shuffle_epi8(batch_ctr, bswap_ctr));
         batch_ctr = _mm256_add_epi32(batch_ctr, inc2);
      }

      // Regenerate the pad buffer
      m_cipher->encrypt_n(m_counter.data(), m_pad.data(), ctr_blocks);
   }

   // NOLINTEND(portability-simd-intrinsics)

   return processed;
}

}  // namespace Botan
