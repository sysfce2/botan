/*
* DESX
* (C) 1999-2007 Jack Lloyd
*
* Distributed under the terms of the Botan license
*/

#ifndef BOTAN_DESX_H__
#define BOTAN_DESX_H__

#include <botan/des.h>

namespace Botan {

/**
* DESX
*/
class BOTAN_DLL DESX : public BlockCipher
   {
   public:
      void encrypt_n(const byte in[], byte out[], size_t blocks) const;
      void decrypt_n(const byte in[], byte out[], size_t blocks) const;

      void clear() { des.clear(); zeroise(K1); zeroise(K2); }
      std::string name() const { return "DESX"; }
      BlockCipher* clone() const { return new DESX; }

      DESX() : BlockCipher(8, 24), K1(8), K2(8) {}
   private:
      void key_schedule(const byte[], u32bit);
      SecureVector<byte> K1, K2;
      DES des;
   };

}

#endif
