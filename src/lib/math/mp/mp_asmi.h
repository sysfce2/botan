/*
* Lowest Level MPI Algorithms
* (C) 1999-2010,2025 Jack Lloyd
*     2006 Luca Piccarreta
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#ifndef BOTAN_MP_ASM_INTERNAL_H_
#define BOTAN_MP_ASM_INTERNAL_H_

#include <botan/compiler.h>
#include <botan/types.h>
#include <botan/internal/target_info.h>
#include <concepts>

#if !defined(BOTAN_TARGET_HAS_NATIVE_UINT128)
   #include <botan/internal/donna128.h>
#endif

namespace Botan {

// NOLINTBEGIN(*-macro-usage)

#if defined(BOTAN_USE_GCC_INLINE_ASM) && defined(BOTAN_TARGET_ARCH_IS_X86_64)
   #define BOTAN_MP_USE_X86_64_ASM
#endif

/*
* Expressing an add with carry is sadly quite difficult in standard C/C++.
*
* Compilers will recognize various idioms and generate a reasonable carry
* chain. Unfortunately which idioms the compiler will understand vary, so we
* have to decide what to do based on the compiler. This is fragile; what will
* work varies not just based on compiler but also version, target architecture,
* and optimization flags.
*/
#if defined(__clang__)
static constexpr bool use_dword_for_word_add = false;
#else
static constexpr bool use_dword_for_word_add = true;
#endif

/*
* Concept for allowed multiprecision word types
*/
template <typename T>
concept WordType = (std::same_as<T, uint32_t> || std::same_as<T, uint64_t>);

template <WordType W>
struct WordInfo {};

template <>
struct WordInfo<uint32_t> {
   public:
      static const constexpr size_t bytes = 4;
      static const constexpr size_t bits = 32;
      static const constexpr uint32_t max = 0xFFFFFFFF;
      static const constexpr uint32_t top_bit = 0x80000000;

      typedef uint64_t dword;
      static const constexpr bool dword_is_native = true;
};

template <>
struct WordInfo<uint64_t> {
   public:
      static const constexpr size_t bytes = 8;
      static const constexpr size_t bits = 64;
      static const constexpr uint64_t max = 0xFFFFFFFFFFFFFFFF;
      static const constexpr uint64_t top_bit = 0x8000000000000000;

#if defined(BOTAN_TARGET_HAS_NATIVE_UINT128)
      typedef uint128_t dword;
      static const constexpr bool dword_is_native = true;
#else
      typedef donna128 dword;
      static const constexpr bool dword_is_native = false;
#endif
};

/*
* Word Multiply/Add
*/
template <WordType W>
inline constexpr auto word_madd2(W a, W b, W* c) -> W {
#if defined(BOTAN_MP_USE_X86_64_ASM)
   if(std::same_as<W, uint64_t> && !std::is_constant_evaluated()) {
      asm(R"(
         mulq %[b]
         addq %[c],%[a]
         adcq $0,%[carry]
         )"
          : [a] "=a"(a), [b] "=rm"(b), [carry] "=&d"(*c)
          : "0"(a), "1"(b), [c] "g"(*c)
          : "cc");

      return a;
   }
#endif

   typedef typename WordInfo<W>::dword dword;
   const dword s = dword(a) * b + *c;
   *c = static_cast<W>(s >> WordInfo<W>::bits);
   return static_cast<W>(s);
}

/*
* Word Multiply/Add
*/
template <WordType W>
inline constexpr auto word_madd3(W a, W b, W c, W* d) -> W {
#if defined(BOTAN_MP_USE_X86_64_ASM)
   if(std::same_as<W, uint64_t> && !std::is_constant_evaluated()) {
      asm(R"(
         mulq %[b]

         addq %[c],%[a]
         adcq $0,%[carry]

         addq %[d],%[a]
         adcq $0,%[carry]
         )"
          : [a] "=a"(a), [b] "=rm"(b), [carry] "=&d"(*d)
          : "0"(a), "1"(b), [c] "g"(c), [d] "g"(*d)
          : "cc");

      return a;
   }
#endif

   typedef typename WordInfo<W>::dword dword;
   const dword s = dword(a) * b + c + *d;
   *d = static_cast<W>(s >> WordInfo<W>::bits);
   return static_cast<W>(s);
}

#if defined(BOTAN_MP_USE_X86_64_ASM)

   #define ASM(x) x "\n\t"

   #define DO_8_TIMES(MACRO, ARG) \
      MACRO(ARG, 0)               \
      MACRO(ARG, 1)               \
      MACRO(ARG, 2)               \
      MACRO(ARG, 3)               \
      MACRO(ARG, 4)               \
      MACRO(ARG, 5)               \
      MACRO(ARG, 6)               \
      MACRO(ARG, 7)

   #define ADDSUB2_OP(OPERATION, INDEX)        \
      ASM("movq 8*" #INDEX "(%[y]), %[carry]") \
      ASM(OPERATION " %[carry], 8*" #INDEX "(%[x])")

   #define ADDSUB3_OP(OPERATION, INDEX)              \
      ASM("movq 8*" #INDEX "(%[x]), %[carry]")       \
      ASM(OPERATION " 8*" #INDEX "(%[y]), %[carry]") \
      ASM("movq %[carry], 8*" #INDEX "(%[z])")

   #define LINMUL_OP(WRITE_TO, INDEX)      \
      ASM("movq 8*" #INDEX "(%[x]),%%rax") \
      ASM("mulq %[y]")                     \
      ASM("addq %[carry],%%rax")           \
      ASM("adcq $0,%%rdx")                 \
      ASM("movq %%rdx,%[carry]")           \
      ASM("movq %%rax, 8*" #INDEX "(%[" WRITE_TO "])")

   #define MULADD_OP(IGNORED, INDEX)       \
      ASM("movq 8*" #INDEX "(%[x]),%%rax") \
      ASM("mulq %[y]")                     \
      ASM("addq %[carry],%%rax")           \
      ASM("adcq $0,%%rdx")                 \
      ASM("addq 8*" #INDEX "(%[z]),%%rax") \
      ASM("adcq $0,%%rdx")                 \
      ASM("movq %%rdx,%[carry]")           \
      ASM("movq %%rax, 8*" #INDEX " (%[z])")

   #define ADD_OR_SUBTRACT(CORE_CODE) \
      ASM("rorq %[carry]")            \
      CORE_CODE                       \
      ASM("sbbq %[carry],%[carry]")   \
      ASM("negq %[carry]")

#endif

/*
* Word Addition
*/
template <WordType W>
inline constexpr auto word_add(W x, W y, W* carry) -> W {
#if BOTAN_COMPILER_HAS_BUILTIN(__builtin_addc)
   if(!std::is_constant_evaluated()) {
      if constexpr(std::same_as<W, unsigned int>) {
         return __builtin_addc(x, y, *carry & 1, carry);
      } else if constexpr(std::same_as<W, unsigned long>) {
         return __builtin_addcl(x, y, *carry & 1, carry);
      } else if constexpr(std::same_as<W, unsigned long long>) {
         return __builtin_addcll(x, y, *carry & 1, carry);
      }
   }
#endif

   if constexpr(WordInfo<W>::dword_is_native && use_dword_for_word_add) {
      /*
      TODO(Botan4) this is largely a performance hack for GCCs that don't
      support __builtin_addc, if we increase the minimum supported version of
      GCC to GCC 14 then we can remove this and not worry about it
      */
      const W cb = *carry & 1;
      const auto s = typename WordInfo<W>::dword(x) + y + cb;
      *carry = static_cast<W>(s >> WordInfo<W>::bits);
      return static_cast<W>(s);
   } else {
      const W cb = *carry & 1;
      W z = x + y;
      W c1 = (z < x);
      z += cb;
      *carry = c1 | (z < cb);
      return z;
   }
}

/*
* Eight Word Block Addition, Two Argument
*/
template <WordType W>
inline constexpr auto word8_add2(W x[8], const W y[8], W carry) -> W {
#if defined(BOTAN_MP_USE_X86_64_ASM)
   if(std::same_as<W, uint64_t> && !std::is_constant_evaluated()) {
      asm volatile(ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB2_OP, "adcq"))
                   : [carry] "=r"(carry)
                   : [x] "r"(x), [y] "r"(y), "0"(carry)
                   : "cc", "memory");
      return carry;
   }
#endif

   x[0] = word_add(x[0], y[0], &carry);
   x[1] = word_add(x[1], y[1], &carry);
   x[2] = word_add(x[2], y[2], &carry);
   x[3] = word_add(x[3], y[3], &carry);
   x[4] = word_add(x[4], y[4], &carry);
   x[5] = word_add(x[5], y[5], &carry);
   x[6] = word_add(x[6], y[6], &carry);
   x[7] = word_add(x[7], y[7], &carry);
   return carry;
}

/*
* Eight Word Block Addition, Three Argument
*/
template <WordType W>
inline constexpr auto word8_add3(W z[8], const W x[8], const W y[8], W carry) -> W {
#if defined(BOTAN_MP_USE_X86_64_ASM)
   if(std::same_as<W, uint64_t> && !std::is_constant_evaluated()) {
      asm volatile(ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB3_OP, "adcq"))
                   : [carry] "=r"(carry)
                   : [x] "r"(x), [y] "r"(y), [z] "r"(z), "0"(carry)
                   : "cc", "memory");
      return carry;
   }
#endif

   z[0] = word_add(x[0], y[0], &carry);
   z[1] = word_add(x[1], y[1], &carry);
   z[2] = word_add(x[2], y[2], &carry);
   z[3] = word_add(x[3], y[3], &carry);
   z[4] = word_add(x[4], y[4], &carry);
   z[5] = word_add(x[5], y[5], &carry);
   z[6] = word_add(x[6], y[6], &carry);
   z[7] = word_add(x[7], y[7], &carry);
   return carry;
}

/*
* Word Subtraction
*/
template <WordType W>
inline constexpr auto word_sub(W x, W y, W* carry) -> W {
#if BOTAN_COMPILER_HAS_BUILTIN(__builtin_subc)
   if(!std::is_constant_evaluated()) {
      if constexpr(std::same_as<W, unsigned int>) {
         return __builtin_subc(x, y, *carry & 1, carry);
      } else if constexpr(std::same_as<W, unsigned long>) {
         return __builtin_subcl(x, y, *carry & 1, carry);
      } else if constexpr(std::same_as<W, unsigned long long>) {
         return __builtin_subcll(x, y, *carry & 1, carry);
      }
   }
#endif

   const W cb = *carry & 1;
   W t0 = x - y;
   W c1 = (t0 > x);
   W z = t0 - cb;
   *carry = c1 | (z > t0);
   return z;
}

/*
* Eight Word Block Subtraction, Two Argument
*/
template <WordType W>
inline constexpr auto word8_sub2(W x[8], const W y[8], W carry) -> W {
#if defined(BOTAN_MP_USE_X86_64_ASM)
   if(std::same_as<W, uint64_t> && !std::is_constant_evaluated()) {
      asm(ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB2_OP, "sbbq"))
          : [carry] "=r"(carry)
          : [x] "r"(x), [y] "r"(y), "0"(carry)
          : "cc", "memory");
      return carry;
   }
#endif

   x[0] = word_sub(x[0], y[0], &carry);
   x[1] = word_sub(x[1], y[1], &carry);
   x[2] = word_sub(x[2], y[2], &carry);
   x[3] = word_sub(x[3], y[3], &carry);
   x[4] = word_sub(x[4], y[4], &carry);
   x[5] = word_sub(x[5], y[5], &carry);
   x[6] = word_sub(x[6], y[6], &carry);
   x[7] = word_sub(x[7], y[7], &carry);
   return carry;
}

/*
* Eight Word Block Subtraction, Three Argument
*/
template <WordType W>
inline constexpr auto word8_sub3(W z[8], const W x[8], const W y[8], W carry) -> W {
#if defined(BOTAN_MP_USE_X86_64_ASM)
   if(std::same_as<W, uint64_t> && !std::is_constant_evaluated()) {
      asm volatile(ADD_OR_SUBTRACT(DO_8_TIMES(ADDSUB3_OP, "sbbq"))
                   : [carry] "=r"(carry)
                   : [x] "r"(x), [y] "r"(y), [z] "r"(z), "0"(carry)
                   : "cc", "memory");
      return carry;
   }
#endif

   z[0] = word_sub(x[0], y[0], &carry);
   z[1] = word_sub(x[1], y[1], &carry);
   z[2] = word_sub(x[2], y[2], &carry);
   z[3] = word_sub(x[3], y[3], &carry);
   z[4] = word_sub(x[4], y[4], &carry);
   z[5] = word_sub(x[5], y[5], &carry);
   z[6] = word_sub(x[6], y[6], &carry);
   z[7] = word_sub(x[7], y[7], &carry);
   return carry;
}

/*
* Eight Word Block Linear Multiplication
*/
template <WordType W>
inline constexpr auto word8_linmul3(W z[8], const W x[8], W y, W carry) -> W {
#if defined(BOTAN_MP_USE_X86_64_ASM)
   if(std::same_as<W, uint64_t> && !std::is_constant_evaluated()) {
      asm(DO_8_TIMES(LINMUL_OP, "z")
          : [carry] "=r"(carry)
          : [z] "r"(z), [x] "r"(x), [y] "rm"(y), "0"(carry)
          : "cc", "%rax", "%rdx");
      return carry;
   }
#endif

   z[0] = word_madd2(x[0], y, &carry);
   z[1] = word_madd2(x[1], y, &carry);
   z[2] = word_madd2(x[2], y, &carry);
   z[3] = word_madd2(x[3], y, &carry);
   z[4] = word_madd2(x[4], y, &carry);
   z[5] = word_madd2(x[5], y, &carry);
   z[6] = word_madd2(x[6], y, &carry);
   z[7] = word_madd2(x[7], y, &carry);
   return carry;
}

/*
* Eight Word Block Multiply/Add
*/
template <WordType W>
inline constexpr auto word8_madd3(W z[8], const W x[8], W y, W carry) -> W {
#if defined(BOTAN_MP_USE_X86_64_ASM)
   if(std::same_as<W, uint64_t> && !std::is_constant_evaluated()) {
      asm(DO_8_TIMES(MULADD_OP, "")
          : [carry] "=r"(carry)
          : [z] "r"(z), [x] "r"(x), [y] "rm"(y), "0"(carry)
          : "cc", "%rax", "%rdx");
      return carry;
   }
#endif

   z[0] = word_madd3(x[0], y, z[0], &carry);
   z[1] = word_madd3(x[1], y, z[1], &carry);
   z[2] = word_madd3(x[2], y, z[2], &carry);
   z[3] = word_madd3(x[3], y, z[3], &carry);
   z[4] = word_madd3(x[4], y, z[4], &carry);
   z[5] = word_madd3(x[5], y, z[5], &carry);
   z[6] = word_madd3(x[6], y, z[6], &carry);
   z[7] = word_madd3(x[7], y, z[7], &carry);
   return carry;
}

/**
* Helper for 3-word accumulators
*
* A number of algorithms especially Comba multiplication and
* Montgomery reduction can take advantage of wide accumulators, which
* consume inputs via addition with outputs extracted from the low
* bits.
*/
template <WordType W>
class word3 final {
#if defined(__BITINT_MAXWIDTH__) && (__BITINT_MAXWIDTH__ >= 3 * 64)

   public:
      constexpr word3() : m_w(0) {}

      inline constexpr void mul(W x, W y) { m_w += static_cast<W3>(x) * y; }

      inline constexpr void mul_x2(W x, W y) { m_w += static_cast<W3>(x) * y * 2; }

      inline constexpr void add(W x) { m_w += x; }

      inline constexpr W extract() {
         W r = static_cast<W>(m_w);
         m_w >>= WordInfo<W>::bits;
         return r;
      }

      inline constexpr W monty_step(W p0, W p_dash) {
         const W w0 = static_cast<W>(m_w);
         const W r = w0 * p_dash;
         mul(r, p0);
         m_w >>= WordInfo<W>::bits;
         return r;
      }

      inline constexpr W monty_step_pdash1() {
         const W r = static_cast<W>(m_w);
         m_w >>= WordInfo<W>::bits;
         m_w += static_cast<W3>(r);
         return r;
      }

   private:
      __extension__ typedef unsigned _BitInt(WordInfo<W>::bits * 3) W3;
      W3 m_w;
#else

   public:
      constexpr word3() : m_w0(0), m_w1(0), m_w2(0) {}

      inline constexpr void mul(W x, W y) {
   #if defined(BOTAN_MP_USE_X86_64_ASM)
         if(std::same_as<W, uint64_t> && !std::is_constant_evaluated()) {
            W z0 = 0, z1 = 0;

            asm("mulq %[y]" : "=a"(z0), "=d"(z1) : "a"(x), [y] "rm"(y) : "cc");

            asm(R"(
                 addq %[z0],%[w0]
                 adcq %[z1],%[w1]
                 adcq $0,%[w2]
                )"
                : [w0] "=r"(m_w0), [w1] "=r"(m_w1), [w2] "=r"(m_w2)
                : [z0] "r"(z0), [z1] "r"(z1), "0"(m_w0), "1"(m_w1), "2"(m_w2)
                : "cc");
            return;
         }
   #endif

         typedef typename WordInfo<W>::dword dword;
         const dword s = dword(x) * y + m_w0;
         W carry = static_cast<W>(s >> WordInfo<W>::bits);
         m_w0 = static_cast<W>(s);
         m_w1 += carry;
         m_w2 += (m_w1 < carry);
      }

      inline constexpr void mul_x2(W x, W y) {
   #if defined(BOTAN_MP_USE_X86_64_ASM)
         if(std::same_as<W, uint64_t> && !std::is_constant_evaluated()) {
            W z0 = 0, z1 = 0;

            asm("mulq %[y]" : "=a"(z0), "=d"(z1) : "a"(x), [y] "rm"(y) : "cc");

            asm(R"(
                 addq %[z0],%[w0]
                 adcq %[z1],%[w1]
                 adcq $0,%[w2]

                 addq %[z0],%[w0]
                 adcq %[z1],%[w1]
                 adcq $0,%[w2]
                   )"
                : [w0] "=r"(m_w0), [w1] "=r"(m_w1), [w2] "=r"(m_w2)
                : [z0] "r"(z0), [z1] "r"(z1), "0"(m_w0), "1"(m_w1), "2"(m_w2)
                : "cc");
            return;
         }
   #endif

         W carry = 0;
         x = word_madd2(x, y, &carry);
         y = carry;

         carry = 0;
         m_w0 = word_add(m_w0, x, &carry);
         m_w1 = word_add(m_w1, y, &carry);
         m_w2 += carry;

         carry = 0;
         m_w0 = word_add(m_w0, x, &carry);
         m_w1 = word_add(m_w1, y, &carry);
         m_w2 += carry;
      }

      inline constexpr void add(W x) {
         constexpr W z = 0;

         W carry = 0;
         m_w0 = word_add(m_w0, x, &carry);
         m_w1 = word_add(m_w1, z, &carry);
         m_w2 += carry;
      }

      inline constexpr W extract() {
         W r = m_w0;
         m_w0 = m_w1;
         m_w1 = m_w2;
         m_w2 = 0;
         return r;
      }

      inline constexpr W monty_step(W p0, W p_dash) {
         W r = m_w0 * p_dash;
         mul(r, p0);
         m_w0 = m_w1;
         m_w1 = m_w2;
         m_w2 = 0;
         return r;
      }

      inline constexpr W monty_step_pdash1() {
         // If p_dash == 1 then p[0] = -1 and everything simplifies
         const W r = m_w0;
         m_w0 += m_w1;
         m_w1 = m_w2 + (m_w0 < m_w1);
         m_w2 = 0;
         return r;
      }

   private:
      W m_w0;
      W m_w1;
      W m_w2;
#endif
};

#if defined(ASM)
   #undef ASM
   #undef DO_8_TIMES
   #undef ADD_OR_SUBTRACT
   #undef ADDSUB2_OP
   #undef ADDSUB3_OP
   #undef LINMUL_OP
   #undef MULADD_OP
#endif

// NOLINTEND(*-macro-usage)

}  // namespace Botan

#endif
