/*
* (C) 2026 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <botan/internal/sm4.h>

#include <botan/mem_ops.h>
#include <botan/internal/isa_extn.h>
#include <botan/internal/simd_4x32.h>
#include <botan/internal/simd_hwaes.h>

namespace Botan {

namespace {

BOTAN_FORCE_INLINE BOTAN_FN_ISA_HWAES SIMD_4x32 sm4_sbox(const SIMD_4x32& x) {
   /*
   * The SM4 sbox is, like the AES sbox, based on inversion in GF(2^8) plus an
   * affine transformation.
   *
   * See
   * - <https://eprint.iacr.org/2022/1154> sections 3.1 and 3.3
   * - <https://github.com/mjosaarinen/sm4ni>
   * - <https://jukivili.kapsi.fi/web/mastersthesis/thesis_final_sRGB_PDFA2b.pdf>
   *   describes a similar approach for implementing Camellia in section 4.4
   */

   constexpr uint64_t pre_a = gfni_matrix(R"(
      0 0 1 1 0 0 1 0
      0 0 0 1 0 1 0 0
      1 0 1 1 1 1 1 0
      1 0 0 1 1 1 0 1
      0 1 0 1 1 0 0 0
      0 1 0 0 0 1 0 0
      0 0 0 0 1 0 1 0
      1 0 1 1 1 0 1 0)");
   constexpr uint8_t pre_c = 0b00111110;

   constexpr uint64_t post_a = gfni_matrix(R"(
      1 1 0 0 1 1 1 1
      1 1 0 1 0 1 0 1
      0 0 1 0 1 1 0 0
      1 0 0 1 0 1 0 1
      0 0 1 0 1 1 1 0
      0 1 1 0 0 1 0 1
      1 0 1 0 1 1 0 1
      1 0 0 1 0 0 0 1)");
   constexpr uint8_t post_c = 0b11010011;

   constexpr auto pre = Gf2AffineTransformation(pre_a, pre_c);
   constexpr auto post = Gf2AffineTransformation::post_sbox(post_a, post_c);

   return post.affine_transform(hw_aes_sbox(pre.affine_transform(x)));
}

BOTAN_FORCE_INLINE BOTAN_FN_ISA_HWAES SIMD_4x32 sm4_f(const SIMD_4x32& x) {
   const auto sx = sm4_sbox(x);
   // L linear transform
   return sx ^ sx.rotl<2>() ^ sx.rotl<10>() ^ sx.rotl<18>() ^ sx.rotl<24>();
}

BOTAN_FORCE_INLINE BOTAN_FN_ISA_HWAES void sm4_hwaes_encrypt_4(const uint8_t ptext[4 * 16],
                                                               uint8_t ctext[4 * 16],
                                                               std::span<const uint32_t> RK) {
   auto B0 = SIMD_4x32::load_be(ptext);
   auto B1 = SIMD_4x32::load_be(ptext + 16);
   auto B2 = SIMD_4x32::load_be(ptext + 32);
   auto B3 = SIMD_4x32::load_be(ptext + 48);

   SIMD_4x32::transpose(B0, B1, B2, B3);

   for(size_t j = 0; j != 8; ++j) {
      const auto K0 = SIMD_4x32::splat(RK[4 * j]);
      const auto K1 = SIMD_4x32::splat(RK[4 * j + 1]);
      const auto K2 = SIMD_4x32::splat(RK[4 * j + 2]);
      const auto K3 = SIMD_4x32::splat(RK[4 * j + 3]);
      B0 ^= sm4_f(B1 ^ B2 ^ B3 ^ K0);
      B1 ^= sm4_f(B2 ^ B3 ^ B0 ^ K1);
      B2 ^= sm4_f(B3 ^ B0 ^ B1 ^ K2);
      B3 ^= sm4_f(B0 ^ B1 ^ B2 ^ K3);
   }

   // SM4 reverses word order
   SIMD_4x32::transpose(B3, B2, B1, B0);

   B3.store_be(ctext);
   B2.store_be(ctext + 16);
   B1.store_be(ctext + 32);
   B0.store_be(ctext + 48);
}

BOTAN_FORCE_INLINE BOTAN_FN_ISA_HWAES void sm4_hwaes_decrypt_4(const uint8_t ctext[4 * 16],
                                                               uint8_t ptext[4 * 16],
                                                               std::span<const uint32_t> RK) {
   auto B0 = SIMD_4x32::load_be(ctext);
   auto B1 = SIMD_4x32::load_be(ctext + 16);
   auto B2 = SIMD_4x32::load_be(ctext + 32);
   auto B3 = SIMD_4x32::load_be(ctext + 48);

   SIMD_4x32::transpose(B0, B1, B2, B3);

   for(size_t j = 0; j != 8; ++j) {
      const auto K0 = SIMD_4x32::splat(RK[32 - (4 * j + 1)]);
      const auto K1 = SIMD_4x32::splat(RK[32 - (4 * j + 2)]);
      const auto K2 = SIMD_4x32::splat(RK[32 - (4 * j + 3)]);
      const auto K3 = SIMD_4x32::splat(RK[32 - (4 * j + 4)]);
      B0 ^= sm4_f(B1 ^ B2 ^ B3 ^ K0);
      B1 ^= sm4_f(B2 ^ B3 ^ B0 ^ K1);
      B2 ^= sm4_f(B3 ^ B0 ^ B1 ^ K2);
      B3 ^= sm4_f(B0 ^ B1 ^ B2 ^ K3);
   }

   // SM4 reverses word order
   SIMD_4x32::transpose(B3, B2, B1, B0);

   B3.store_be(ptext);
   B2.store_be(ptext + 16);
   B1.store_be(ptext + 32);
   B0.store_be(ptext + 48);
}

}  // namespace

void BOTAN_FN_ISA_HWAES SM4::sm4_hwaes_encrypt(const uint8_t ptext[], uint8_t ctext[], size_t blocks) const {
   while(blocks >= 4) {
      sm4_hwaes_encrypt_4(ptext, ctext, m_RK);
      ptext += 16 * 4;
      ctext += 16 * 4;
      blocks -= 4;
   }

   if(blocks > 0) {
      uint8_t pbuf[4 * 16] = {0};
      uint8_t cbuf[4 * 16] = {0};
      copy_mem(pbuf, ptext, blocks * 16);
      sm4_hwaes_encrypt_4(pbuf, cbuf, m_RK);
      copy_mem(ctext, cbuf, blocks * 16);
   }
}

void BOTAN_FN_ISA_HWAES SM4::sm4_hwaes_decrypt(const uint8_t ctext[], uint8_t ptext[], size_t blocks) const {
   while(blocks >= 4) {
      sm4_hwaes_decrypt_4(ctext, ptext, m_RK);
      ptext += 16 * 4;
      ctext += 16 * 4;
      blocks -= 4;
   }

   if(blocks > 0) {
      uint8_t cbuf[4 * 16] = {0};
      uint8_t pbuf[4 * 16] = {0};
      copy_mem(cbuf, ctext, blocks * 16);
      sm4_hwaes_decrypt_4(cbuf, pbuf, m_RK);
      copy_mem(ptext, pbuf, blocks * 16);
   }
}

}  // namespace Botan
