#include <botan/hex.h>
#include <botan/internal/loadstor.h>
#include <botan/internal/mp_core.h>
#include <botan/internal/stl_util.h>
#include <botan/internal/ec_h2c.h>
#include <botan/rng.h>
#include <array>
#include <optional>
#include <iostream>

namespace Botan {

template <size_t N>
void dump(const char* s, const std::array<uint64_t, N>& x) {
   printf("%s [%lu] = ", s, N);
   for(size_t i = 0; i != N; ++i) {
      printf("%016lX ", x.at(N - i - 1));
   }
   printf("\n");
}

template <size_t N>
class StringLiteral {
   public:
      constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }

      char value[N];
};

template <WordType W, size_t N, size_t XN>
inline consteval std::array<W, N> reduce_mod(const std::array<W, XN>& x, const std::array<W, N>& p) {
   std::array<W, N + 1> r = {0};
   std::array<W, N + 1> t = {0};

   const size_t x_bits = XN * WordInfo<W>::bits;

   for(size_t i = 0; i != x_bits; ++i) {
      const size_t b = x_bits - 1 - i;

      const size_t b_word = b / WordInfo<W>::bits;
      const size_t b_bit = b % WordInfo<W>::bits;
      const bool x_b = (x[b_word] >> b_bit) & 1;

      shift_left<1>(r);
      if(x_b) {
         r[0] += 1;
      }

      const W carry = bigint_sub3(t.data(), r.data(), N + 1, p.data(), N);

      if(carry == 0) {
         std::swap(r, t);
      }
   }

   std::array<W, N> rs;
   std::copy(r.begin(), r.begin() + N, rs.begin());
   return rs;
}

template <WordType W, size_t N>
inline consteval std::array<W, N> montygomery_r(const std::array<W, N>& p) {
   std::array<W, N + 1> x = {0};
   x[N] = 1;
   return reduce_mod(x, p);
}

template <WordType W, size_t N>
inline consteval std::array<W, N> mul_mod(const std::array<W, N>& x,
                                          const std::array<W, N>& y,
                                          const std::array<W, N>& p) {
   std::array<W, 2 * N> z;
   comba_mul<N>(z.data(), x.data(), y.data());
   return reduce_mod(z, p);
}

template <WordType W, size_t N, size_t ZL>
inline constexpr auto bigint_monty_redc(const std::array<W, ZL>& z, const std::array<W, N>& p, word p_dash)
   -> std::array<W, N> {
   static_assert(N >= 1);
   static_assert(ZL <= 2 * N);

   std::array<W, N> ws;

   W w2 = 0, w1 = 0, w0 = 0;

   w0 = z[0];

   ws[0] = w0 * p_dash;

   word3_muladd(&w2, &w1, &w0, ws[0], p[0]);

   w0 = w1;
   w1 = w2;
   w2 = 0;

   for(size_t i = 1; i != N; ++i) {
      for(size_t j = 0; j < i; ++j) {
         word3_muladd(&w2, &w1, &w0, ws[j], p[i - j]);
      }

      word3_add(&w2, &w1, &w0, i < ZL ? z[i] : 0);

      ws[i] = w0 * p_dash;

      word3_muladd(&w2, &w1, &w0, ws[i], p[0]);

      w0 = w1;
      w1 = w2;
      w2 = 0;
   }

   for(size_t i = 0; i != N - 1; ++i) {
      for(size_t j = i + 1; j != N; ++j) {
         word3_muladd(&w2, &w1, &w0, ws[j], p[N + i - j]);
      }

      word3_add(&w2, &w1, &w0, N + i < ZL ? z[N + i] : 0);

      ws[i] = w0;

      w0 = w1;
      w1 = w2;
      w2 = 0;
   }

   word3_add(&w2, &w1, &w0, (2 * N - 1) < ZL ? z[2 * N - 1] : 0);

   ws[N - 1] = w0;

   std::array<W, N> r = {0};
   for(size_t i = 0; i != std::min(ZL, N); ++i) {
      r[i] = z[i];
   }
   bigint_monty_maybe_sub<N>(r.data(), w1, ws.data(), p.data());

   return r;
}

template <uint8_t X, WordType W, size_t N>
inline consteval std::array<W, N> p_minus(const std::array<W, N>& p) {
   static_assert(X > 0);
   std::array<W, N> r;
   W x = X;
   bigint_sub3(r.data(), p.data(), N, &x, 1);
   return r;
}

template <WordType W, size_t N>
inline constexpr uint8_t get_bit(size_t i, const std::array<W, N>& p) {
   const size_t w = i / WordInfo<W>::bits;
   const size_t b = i % WordInfo<W>::bits;

   return static_cast<uint8_t>((p[w] >> b) & 0x01);
}

template <WordType W, size_t N>
inline consteval size_t count_bits(const std::array<W, N>& p) {
   size_t b = WordInfo<W>::bits * N;

   while(get_bit(b - 1, p) == 0) {
      b -= 1;
   }

   return b;
}

template <WordType W, size_t N, size_t L>
inline constexpr auto bytes_to_words(std::span<const uint8_t, L> n) {
   static_assert(L <= WordInfo<W>::bytes * N);

   std::array<W, N> r = {};
   for(size_t i = 0; i != L; ++i) {
      shift_left<8>(r);
      r[0] += n[i];
   }
   return r;
}

template <StringLiteral PS>
class MontgomeryInteger {
   private:
      typedef word W;

      static const constexpr auto P = hex_to_words<W>(PS.value);
      static const constexpr size_t N = P.size();

      // One can dream
      //static_assert(is_prime(P), "Montgomery Modulus must be a prime");
      static_assert(N > 0 && (P[0] & 1) == 1, "Invalid Montgomery modulus");

      static const constexpr W P_dash = monty_inverse(P[0]);

      static const constexpr auto R1 = montygomery_r(P);
      static const constexpr auto R2 = mul_mod(R1, R1, P);
      static const constexpr auto R3 = mul_mod(R1, R2, P);

      static const constexpr auto P_MINUS_2 = p_minus<2>(P);

   public:
      static const constexpr size_t BITS = count_bits(P);
      static const constexpr size_t BYTES = (BITS + 7) / 8;

      typedef MontgomeryInteger<PS> Self;

      // Default value is zero
      constexpr MontgomeryInteger() : m_val({}) {}

      MontgomeryInteger(const Self& other) = default;
      MontgomeryInteger(Self&& other) = default;
      MontgomeryInteger& operator=(const Self& other) = default;
      MontgomeryInteger& operator=(Self&& other) = default;

      // ??
      //~MontgomeryInteger() { secure_scrub_memory(m_val); }

      static constexpr Self zero() { return Self(std::array<W, N>{0}); }

      static constexpr Self one() { return Self(Self::R1); }

      constexpr bool is_zero() const { return CT::all_zeros(m_val.data(), m_val.size()).as_bool(); }

      constexpr bool is_one() const { return (*this == Self::one()); }

      // This happens to work without converting from Montgomery form
      constexpr bool is_even() const { return (m_val[0] & 0x01) == 0x00; }

      friend constexpr Self operator+(const Self& a, const Self& b) {
         std::array<W, N> t;
         W carry = bigint_add3_nc(t.data(), a.data(), N, b.data(), N);

         std::array<W, N> r;
         bigint_monty_maybe_sub<N>(r.data(), carry, t.data(), Self::P.data());
         return Self(r);
      }

      constexpr Self& operator+=(const Self& other) {
         std::array<W, N> t;
         W carry = bigint_add3_nc(t.data(), this->data(), N, other.data(), N);
         bigint_monty_maybe_sub<N>(m_val.data(), carry, t.data(), Self::P.data());
         return (*this);
      }

      friend constexpr Self operator-(const Self& a, const Self& b) { return a + b.negate(); }

      friend constexpr Self operator*(uint8_t a, const Self& b) {
         return b * a;
      }

      friend constexpr Self operator*(const Self& a, uint8_t b) {
         // We assume b is a small constant and allow variable time
         // computation

         Self z = Self::zero();
         Self x = a;

         while(b > 0) {
            if(b & 1) {
               z = z + x;
            }
            x = x.dbl();
            b >>= 1;
         }

         return z;
      }

      friend constexpr Self operator*(const Self& a, const Self& b) {
         std::array<W, 2 * N> z;
         comba_mul<N>(z.data(), a.data(), b.data());
         return Self(bigint_monty_redc(z, Self::P, Self::P_dash));
      }

      constexpr void conditional_add(bool cond, const Self& other) { conditional_assign(cond, *this + other); }

      constexpr void conditional_mul(bool cond, const Self& other) { conditional_assign(cond, *this * other); }

      constexpr void conditional_sub(bool cond, const Self& other) { conditional_add(cond, other.negate()); }

      constexpr void conditional_assign(bool cond, const Self& other) {
         CT::conditional_assign_mem(static_cast<W>(cond), m_val.data(), other.data(), N);
      }

      // fixme be faster
      constexpr Self dbl() const { return (*this) + (*this); }

      constexpr Self square() const {
         std::array<W, 2 * N> z;
         comba_sqr<N>(z.data(), this->data());
         return bigint_monty_redc(z, Self::P, Self::P_dash);
      }

      // Negation modulo p
      constexpr Self negate() const {
         auto x_is_zero = CT::all_zeros(this->data(), N);

         std::array<W, N> r;
         bigint_sub3(r.data(), Self::P.data(), N, this->data(), N);
         x_is_zero.if_set_zero_out(r.data(), N);
         return Self(r);
      }

      /**
      * Returns the modular inverse, or 0 if no modular inverse exists.
      *
      * If the modulus is prime the only value that has no modular inverse is 0.
      *
      * This uses Fermat's little theorem, and so assumes that p is prime
      */
      constexpr Self invert() const {
         auto x = (*this);
         auto y = Self::one();

         for(size_t i = 0; i != Self::BITS; ++i) {
            auto b = get_bit(i, P_MINUS_2);
            y.conditional_mul(b, x);
            x = x.square();
         }

         return y;
      }

      constexpr bool operator==(const Self& other) const { return CT::is_equal(this->data(), other.data(), N).as_bool(); }

      constexpr bool operator!=(const Self& other) const {
         return CT::is_not_equal(this->data(), other.data(), N).as_bool();
      }

      constexpr std::array<uint8_t, Self::BYTES> serialize() const {
         auto v = bigint_monty_redc(m_val, Self::P, Self::P_dash);
         std::reverse(v.begin(), v.end());
         return store_be(v);
      }

      // TODO:

      // Returns nullopt if the input is an encoding greater than or equal P
      constexpr static std::optional<Self> deserialize(std::span<const uint8_t, Self::BYTES> bytes) {
         const auto words = bytes_to_words<W, N, BYTES>(bytes);

         // TODO range check!!

         return Self(words) * Self::R2;
      }

      template <size_t L>
      static constexpr Self from_wide_bytes(std::span<const uint8_t, L> bytes) {
         static_assert(L <= 2*Self::BYTES);

      }

      /*
      static Self from_bigint(const BigInt& bn) {

      }

      BigInt to_bigint() const {

      }

      static constexpr Self ct_select(std::span<const Self> several, size_t idx) {

      }
      */

      static constexpr Self random(RandomNumberGenerator& rng) {
         std::array<uint8_t, Self::BYTES> buf;
         for(;;) {
            rng.randomize(buf.data(), buf.size());
            if(auto v = Self::deserialize(buf)) {
               return v;
            }
         }
      }


      template <size_t N>
      static consteval Self constant(StringLiteral<N> S) {
         return Self::constant(S.value);
      }

      template <size_t N>
      static consteval Self constant(const char (&s)[N]) {
         const auto v = hex_to_words<W>(s);
         return Self(v) * R2;
      }

      static consteval Self constant(int8_t x) {
         std::array<W, 1> v;
         v[0] = (x >= 0) ? x : -x;
         auto s = Self(v) * R2;
         return (x >= 0) ? s : s.negate();
      }

   private:
      constexpr const std::array<W, N>& value() const { return m_val; }

      constexpr const W* data() const { return m_val.data(); }

      template<size_t S>
      constexpr MontgomeryInteger(std::array<W, S> w) : m_val({}) {
         static_assert(S <= N);
         for(size_t i = 0; i != S; ++i) {
            m_val[i] = w[i];
         }
      }

      std::array<W, N> m_val;
};

template <StringLiteral PS>
void dump(const char* what, const MontgomeryInteger<PS>& fe) {
   std::cout << what << " = " << hex_encode(fe.serialize()) << "\n";
}

template <typename FieldElement>
class AffineCurvePoint {
   public:
      static const constinit size_t BYTES = 1 + 2 * FieldElement::BYTES;
      static const constinit size_t COMPRESSED_BYTES = 1 + FieldElement::BYTES;

      typedef AffineCurvePoint<FieldElement> Self;

      constexpr AffineCurvePoint(const FieldElement& x, const FieldElement& y) : m_x(x), m_y(y) {}

      constexpr AffineCurvePoint() : m_x(FieldElement::zero()), m_y(FieldElement::zero()) {}

      /*
      y**2 = x**3 + a*x + b

      y**2 - b = x**3 + a*x
      y**2 - b = (x**2 + a)*x
      */
      // ??????
      static constexpr Self identity() { return Self(FieldElement::zero(), FieldElement::zero()); }

      constexpr bool is_identity() const { return m_x.is_zero() && m_y.is_zero(); }

      AffineCurvePoint(const Self& other) = default;
      AffineCurvePoint(Self&& other) = default;
      AffineCurvePoint& operator=(const Self& other) = default;
      AffineCurvePoint& operator=(Self&& other) = default;

      constexpr Self negate() const { return Self(m_x, m_y.negate()); }

      constexpr std::array<uint8_t, Self::BYTES> serialize() const {
         std::array<uint8_t, Self::BYTES> r = {};
         BufferStuffer pack(r);
         pack.append(0x04);
         pack.append(m_x.serialize());
         pack.append(m_y.serialize());

         return r;
      }

      constexpr std::array<uint8_t, Self::COMPRESSED_BYTES> serialize_compressed() const {
         std::array<uint8_t, Self::COMPRESSED_BYTES> r = {};

         const bool y_is_even = y().is_even();

         BufferStuffer pack(r);
         pack.append(y_is_even ? 0x02 : 0x03);
         pack.append(x().serialize());

         return r;
      }

      //static constexpr std::optional<Self> deserialize(std::span<const uint8_t> bytes) {}

      constexpr const FieldElement& x() const { return m_x; }

      constexpr const FieldElement& y() const { return m_y; }

   private:
      FieldElement m_x;
      FieldElement m_y;
};

template <typename FieldElement, StringLiteral AS>
class ProjectiveCurvePoint {
   public:
      // We can't pass a FieldElement directly because FieldElement is
      // not "structural" due to having private members, so instead
      // recreate it here from the string.
      static const constexpr FieldElement A = FieldElement::constant(AS);

      static const constinit bool A_is_minus_3 = (A == FieldElement::constant(-3));
      static const constinit bool A_is_zero = (A == FieldElement::constant(0));

      typedef ProjectiveCurvePoint<FieldElement, AS> Self;
      typedef AffineCurvePoint<FieldElement> AffinePoint;

      static constexpr Self from_affine(const AffinePoint& pt) { return ProjectiveCurvePoint(pt.x(), pt.y()); }

      static constexpr Self identity() {
         return Self(FieldElement::zero(), FieldElement::zero(), FieldElement::zero());
      }

      constexpr ProjectiveCurvePoint() :
         m_x(FieldElement::zero()), m_y(FieldElement::zero()), m_z(FieldElement::zero()) {}

      constexpr ProjectiveCurvePoint(const FieldElement& x, const FieldElement& y) :
            m_x(x), m_y(y), m_z(FieldElement::one()) {}

      constexpr ProjectiveCurvePoint(const FieldElement& x, const FieldElement& y, const FieldElement& z) :
            m_x(x), m_y(y), m_z(z) {}

      ProjectiveCurvePoint(const Self& other) = default;
      ProjectiveCurvePoint(Self&& other) = default;
      ProjectiveCurvePoint& operator=(const Self& other) = default;
      ProjectiveCurvePoint& operator=(Self&& other) = default;

      friend constexpr Self operator+(const Self& a, const Self& b) { return Self::add(a, b); }

      friend constexpr Self operator+(const Self& a, const AffinePoint& b) { return Self::add_mixed(a, b); }

      friend constexpr Self operator+(const AffinePoint& a, const Self& b) { return Self::add_mixed(b, a); }

      constexpr Self& operator+=(const Self& other) {
         (*this) = (*this) + other;
         return (*this);
      }

      constexpr Self& operator+=(const AffinePoint& other) {
         (*this) = (*this) + other;
         return (*this);
      }

      friend constexpr Self operator-(const Self& a, const Self& b) { return a + b.negate(); }

      constexpr bool is_identity() const { return z().is_zero(); }

      template<typename Pt>
      constexpr void conditional_add(bool cond, const Pt& pt) {
         conditional_assign(cond, *this + pt);
      }

      void conditional_assign(bool cond, const Self& pt) {
         m_x.conditional_assign(cond, pt.x());
         m_y.conditional_assign(cond, pt.y());
         m_z.conditional_assign(cond, pt.z());
      }

      constexpr static Self add_mixed(const Self& a, const AffinePoint& b) {
         // fixme use proper mixed addition formula
         return Self::add(a, Self::from_affine(b));
      }

      constexpr static Self add(const Self& a, const Self& b) {
         //printf("add %d %d\n", a.is_identity(), b.is_identity());

         // TODO avoid these early returns by masking instead
         if(a.is_identity()) {
            return b;
         }

         if(b.is_identity()) {
            return a;
         }

         /*
         https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#addition-add-1998-cmo-2

         TODO detect a doubling!! DONE

         TODO rename these vars

         TODO reduce vars

         TODO take advantage of A = 0 and A = -3

         TODO use a complete addition formula??? (YES)
         https://eprint.iacr.org/2015/1060.pdf
         */

         const auto Z1Z1 = a.z().square();
         const auto Z2Z2 = b.z().square();
         const auto U1 = a.x() * Z2Z2;
         const auto U2 = b.x() * Z1Z1;
         const auto S1 = a.y() * b.z() * Z2Z2;
         const auto S2 = b.y() * a.z() * Z1Z1;
         const auto H = U2 - U1;
         const auto r = S2 - S1;

         if(H.is_zero()) {
            if(r.is_zero()) {
               return a.dbl();
            } else {
               return Self::identity();
            }
         }

         const auto HH = H.square();
         const auto HHH = H * HH;
         const auto V = U1 * HH;
         const auto t2 = r.square();
         const auto t3 = V + V;
         const auto t4 = t2 - HHH;
         const auto X3 = t4 - t3;
         const auto t5 = V - X3;
         const auto t6 = S1 * HHH;
         const auto t7 = r * t5;
         const auto Y3 = t7 - t6;
         const auto t8 = b.z() * H;
         const auto Z3 = a.z() * t8;

         return Self(X3, Y3, Z3);
      }

      constexpr Self dbl() const {
         //https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-1998-cmo-2

         FieldElement m = FieldElement::zero();

         if constexpr(Self::A_is_minus_3) {
            /*
            if a == -3 then
            3*x^2 + a*z^4 == 3*x^2 - 3*z^4 == 3*(x^2-z^4) == 3*(x-z^2)*(x+z^2)

            Cost: 2M + 2A + 1*3
            */
            const auto z2 = z().square();
            m = 3 * (x() - z2) * (x() + z2);
         } else if constexpr(Self::A_is_zero) {
            // If a == 0 then 3*x^2 + a*z^4 == 3*x^2
            // Cost: 1S + 1*3
            m = 3 * x().square();
         } else {
            // Cost: 1M + 3S + 1A + 1*3
            const auto z2 = z().square();
            m = 3 * x().square() + A * z2.square();
         }

         const auto y2 = y().square();
         const auto s = 4 * x() * y2;
         const auto nx = m.square() - 2 * s;
         const auto ny = m * (s - nx) - 8 * y2.square();
         const auto nz = 2 * y() * z();

         return Self(nx, ny, nz);
      }

      constexpr Self negate() const { return Self(m_x, m_y.negate(), m_z); }

      constexpr AffinePoint to_affine() const {
         // Not strictly required right? - default should work as long
         // as (0,0) is identity and invert returns 0 on 0
         if(this->is_identity()) {
            return AffinePoint::identity();
         }

         // Maybe also worth skipping ...
         if(m_z.is_one()) {
            return AffinePoint(m_x, m_y);
         }

         const auto z_inv = m_z.invert();
         const auto z2_inv = z_inv.square();
         const auto z3_inv = z_inv * z2_inv;

         const auto x = m_x * z2_inv;
         const auto y = m_y * z3_inv;
         return AffinePoint(x, y);
      }

      template <size_t N>
      static constexpr auto to_affine_batch(const std::array<Self, N>& projective) -> std::array<AffinePoint, N> {
         std::array<AffinePoint, N> affine;

         bool any_identity = false;
         for(size_t i = 0; i != N; ++i) {
            if(projective[i].is_identity()) {
               any_identity = true;
            }
         }

         if(N <= 2 || any_identity) {
            for(size_t i = 0; i != N; ++i) {
               affine[i] = projective[i].to_affine();
            }
         } else {
            std::array<FieldElement, N> c;

            /*
            Batch projective->affine using Montgomery's trick

            See Algorithm 2.26 in "Guide to Elliptic Curve Cryptography"
            (Hankerson, Menezes, Vanstone)
            */

            c[0] = projective[0].z();
            for(size_t i = 1; i != N; ++i) {
               c[i] = c[i - 1] * projective[i].z();
            }

            auto s_inv = c[N-1].invert();

            for(size_t i = N - 1; i > 0; --i) {
               const auto& p = projective[i];

               const auto z_inv = s_inv * c[i - 1];
               const auto z2_inv = z_inv.square();
               const auto z3_inv = z_inv * z2_inv;

               s_inv = s_inv * p.z();

               affine[i] = AffinePoint(p.x() * z2_inv, p.y() * z3_inv);
            }

            const auto z2_inv = s_inv.square();
            const auto z3_inv = s_inv * z2_inv;
            affine[0] = AffinePoint(projective[0].x() * z2_inv,
                                    projective[0].y() * z3_inv);
         }

         return affine;
      }

      constexpr const FieldElement& x() const { return m_x; }

      constexpr const FieldElement& y() const { return m_y; }

      constexpr const FieldElement& z() const { return m_z; }

   private:
      FieldElement m_x;
      FieldElement m_y;
      FieldElement m_z;
};

template <typename AffinePoint, typename ProjectivePoint, typename Scalar>
class PrecomputedMulTable {
   public:
      //static const constinit WINDOW_BITS = 1; // XXX allow config?

      //static_assert(WINDOW_BITS >= 1 && WINDOW_BITS <= 8);

      static const constinit size_t TABLE_SIZE = Scalar::BITS;

      constexpr PrecomputedMulTable(const AffinePoint& p) : m_table{} {
         std::array<ProjectivePoint, TABLE_SIZE> table;

         table[0] = ProjectivePoint::from_affine(p);
         for(size_t i = 1; i != TABLE_SIZE; ++i) {
            table[i] = table[i-1].dbl();
         }

         m_table = ProjectivePoint::to_affine_batch(table);
      }

      constexpr AffinePoint operator()(const Scalar& s) const {
         const auto bits = s.serialize();

         auto accum = ProjectivePoint::identity();

         for(size_t i = 0; i != Scalar::BITS; ++i) {
            const size_t b = Scalar::BITS - i - 1;
            const bool b_set = (bits[b / 8] >> (7 - b % 8)) & 1;
            accum.conditional_add(b_set, m_table[i]);
         }

         return accum.to_affine();
      }

   private:
      std::array<AffinePoint, TABLE_SIZE> m_table;
};

template <StringLiteral PS,
          StringLiteral AS,
          StringLiteral BS,
          StringLiteral NS,
          StringLiteral GXS,
          StringLiteral GYS,
          template <StringLiteral> typename FieldType = MontgomeryInteger>
class EllipticCurve {
   public:
      typedef MontgomeryInteger<NS> Scalar;
      typedef FieldType<PS> FieldElement;

      static const constexpr FieldElement A = FieldElement::constant(AS);
      static const constexpr FieldElement B = FieldElement::constant(BS);
      static const constexpr FieldElement Gx = FieldElement::constant(GXS);
      static const constexpr FieldElement Gy = FieldElement::constant(GYS);

      typedef AffineCurvePoint<FieldElement> AffinePoint;
      typedef ProjectiveCurvePoint<FieldElement, AS> ProjectivePoint;

      static const constexpr AffinePoint G = AffinePoint(Gx, Gy);

      typedef PrecomputedMulTable<AffinePoint, ProjectivePoint, Scalar> MulTable;
};


template <typename C>
C::AffinePoint scalar_mul(const typename C::AffinePoint& p,
                          const typename C::Scalar& s) {
   const auto bits = s.serialize();

   auto accum = C::ProjectivePoint::identity();
   auto pp = C::ProjectivePoint::from_affine(p);

   for(size_t i = 0; i != C::Scalar::BITS; ++i) {
      const size_t b = C::Scalar::BITS - i - 1;

      const bool b_set = (bits[b / 8] >> (7 - b % 8)) & 1;

      //accum.conditional_add(b_set, pp);
      if(b_set) {
         accum = accum + pp;
      }
      pp = pp.dbl();
   }

   return accum.to_affine();
}

}  // namespace Botan

int main() {
   using namespace Botan;

   typedef MontgomeryInteger<"FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF"> fe;

   static_assert(fe::BITS == 256);
   static_assert(fe::BYTES == 32);

   #if 1
   auto s = fe::one();
   for(size_t i = 1; i != 30; ++i) {
      auto bytes = s.serialize();
      std::cout << hex_encode(bytes) << "\n";

      auto r = fe::deserialize(bytes);
      if(s != r) {
         if(r) {
            std::cout << hex_encode(r.value().serialize()) << "\n";
         } else {
            std::cout << "(none)\n";
         }
      }
      s = s - fe::one();
   }
   #endif

#if 0
   typedef EllipticCurve<"FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF",
                         "FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC",
                         "5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B",
                         "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
                         "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
                         "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5">
      P256;


   #if 1
   typedef EllipticCurve<"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
      "0",
      "7",
      "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
      "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
      "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8">
      K256;

   K256::MulTable K256_mul(K256::G);

   auto s = K256::Scalar::zero();

   for(size_t i = 0; i != 10; ++i) {
      auto p = K256_mul(s);
      //auto p = scalar_mul<P256>(P256::G, s);
      std::cout << i << " -> " << hex_encode(p.serialize()) << "\n";
      s = s + K256::Scalar::one();
   }
   #endif
   #endif
}
