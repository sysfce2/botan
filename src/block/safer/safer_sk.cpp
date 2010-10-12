/*
* SAFER-SK
* (C) 1999-2009 Jack Lloyd
*
* Distributed under the terms of the Botan license
*/

#include <botan/safer_sk.h>
#include <botan/rotate.h>
#include <botan/parsing.h>
#include <botan/rotate.h>

namespace Botan {

/*
* SAFER-SK Encryption
*/
void SAFER_SK::encrypt_n(const byte in[], byte out[], size_t blocks) const
   {
   for(size_t i = 0; i != blocks; ++i)
      {
      byte A = in[0], B = in[1], C = in[2], D = in[3],
           E = in[4], F = in[5], G = in[6], H = in[7], X, Y;

      for(size_t j = 0; j != 16*ROUNDS; j += 16)
         {
         A = EXP[A ^ EK[j  ]]; B = LOG[B + EK[j+1]];
         C = LOG[C + EK[j+2]]; D = EXP[D ^ EK[j+3]];
         E = EXP[E ^ EK[j+4]]; F = LOG[F + EK[j+5]];
         G = LOG[G + EK[j+6]]; H = EXP[H ^ EK[j+7]];

         A += EK[j+ 8]; B ^= EK[j+ 9]; C ^= EK[j+10]; D += EK[j+11];
         E += EK[j+12]; F ^= EK[j+13]; G ^= EK[j+14]; H += EK[j+15];

         B += A; D += C; F += E; H += G; A += B; C += D; E += F; G += H;
         C += A; G += E; D += B; H += F; A += C; E += G; B += D; F += H;
         H += D; Y = D + H; D = B + F; X = B + D; B = A + E;
         A += B; F = C + G; E = C + F; C = X; G = Y;
         }

      out[0] = A ^ EK[16*ROUNDS+0]; out[1] = B + EK[16*ROUNDS+1];
      out[2] = C + EK[16*ROUNDS+2]; out[3] = D ^ EK[16*ROUNDS+3];
      out[4] = E ^ EK[16*ROUNDS+4]; out[5] = F + EK[16*ROUNDS+5];
      out[6] = G + EK[16*ROUNDS+6]; out[7] = H ^ EK[16*ROUNDS+7];

      in += BLOCK_SIZE;
      out += BLOCK_SIZE;
      }
   }

/*
* SAFER-SK Decryption
*/
void SAFER_SK::decrypt_n(const byte in[], byte out[], size_t blocks) const
   {
   for(size_t i = 0; i != blocks; ++i)
      {
      byte A = in[0], B = in[1], C = in[2], D = in[3],
           E = in[4], F = in[5], G = in[6], H = in[7];

      A ^= EK[16*ROUNDS+0]; B -= EK[16*ROUNDS+1]; C -= EK[16*ROUNDS+2];
      D ^= EK[16*ROUNDS+3]; E ^= EK[16*ROUNDS+4]; F -= EK[16*ROUNDS+5];
      G -= EK[16*ROUNDS+6]; H ^= EK[16*ROUNDS+7];

      for(s32bit j = 16*(ROUNDS-1); j >= 0; j -= 16)
         {
         byte T = E; E = B; B = C; C = T; T = F; F = D; D = G; G = T;
         A -= E; B -= F; C -= G; D -= H; E -= A; F -= B; G -= C; H -= D;
         A -= C; E -= G; B -= D; F -= H; C -= A; G -= E; D -= B; H -= F;
         A -= B; C -= D; E -= F; G -= H; B -= A; D -= C; F -= E; H -= G;

         A = LOG[A - EK[j+8 ] + 256]; B = EXP[B ^ EK[j+9 ]];
         C = EXP[C ^ EK[j+10]];       D = LOG[D - EK[j+11] + 256];
         E = LOG[E - EK[j+12] + 256]; F = EXP[F ^ EK[j+13]];
         G = EXP[G ^ EK[j+14]];       H = LOG[H - EK[j+15] + 256];

         A ^= EK[j+0]; B -= EK[j+1]; C -= EK[j+2]; D ^= EK[j+3];
         E ^= EK[j+4]; F -= EK[j+5]; G -= EK[j+6]; H ^= EK[j+7];
         }

      out[0] = A; out[1] = B; out[2] = C; out[3] = D;
      out[4] = E; out[5] = F; out[6] = G; out[7] = H;

      in += BLOCK_SIZE;
      out += BLOCK_SIZE;
      }
   }

/*
* SAFER-SK Key Schedule
*/
void SAFER_SK::key_schedule(const byte key[], u32bit)
   {
   SecureVector<byte> KB(18);

   for(size_t i = 0; i != 8; ++i)
      {
      KB[ 8] ^= KB[i] = rotate_left(key[i], 5);
      KB[17] ^= KB[i+9] = EK[i] = key[i+8];
      }

   for(size_t i = 0; i != ROUNDS; ++i)
      {
      for(size_t j = 0; j != 18; ++j)
         KB[j] = rotate_left(KB[j], 6);
      for(size_t j = 0; j != 16; ++j)
         EK[16*i+j+8] = KB[KEY_INDEX[16*i+j]] + BIAS[16*i+j];
      }
   }

/*
* Return the name of this type
*/
std::string SAFER_SK::name() const
   {
   return "SAFER-SK(" + to_string(ROUNDS) + ")";
   }

/*
* Return a clone of this object
*/
BlockCipher* SAFER_SK::clone() const
   {
   return new SAFER_SK(ROUNDS);
   }

/*
* SAFER-SK Constructor
*/
SAFER_SK::SAFER_SK(size_t rounds) : BlockCipher(8, 16),
                                    EK(16 * rounds + 8), ROUNDS(rounds)
   {
   if(ROUNDS > 13 || ROUNDS == 0)
      throw Invalid_Argument(name() + ": Invalid number of rounds");
   }

}
