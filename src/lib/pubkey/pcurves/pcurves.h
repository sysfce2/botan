/*
* (C) 2024 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#ifndef BOTAN_PCURVES_H_
#define BOTAN_PCURVES_H_

#include <botan/types.h>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace Botan {

class RandomNumberGenerator;
class OID;

}

namespace Botan::PCurve {

/// Identifier for a named prime order curve
class PrimeOrderCurveId {
   public:
      enum class Id : uint8_t {
         /// secp256r1 aka P-256
         secp256r1,
         /// secp384r1 aka P-384
         secp384r1,
         /// secp521r1 aka P-521
         secp521r1,
         /// secp256k1
         secp256k1,
      };

      using enum Id;

      Id code() const { return m_id; }

      PrimeOrderCurveId(Id id) : m_id(id) {}

      /// Map a string to a curve identifier
      BOTAN_TEST_API
      static std::optional<PrimeOrderCurveId> from_string(std::string_view name);

      /// Map an OID to a curve identifier
      ///
      /// Uses the internal OID table
      static std::optional<PrimeOrderCurveId> from_oid(const OID& oid);

      std::string to_string() const;

   private:
      const Id m_id;
};

class PrimeOrderCurve {
   public:
      /// Somewhat arbitrary maximum size for a field or scalar
      ///
      /// Sized to fit at least P-521
      static const size_t MaximumBitLength = 521;

      static const size_t MaximumByteLength = (MaximumBitLength + 7) / 8;

      /// Maximum number of words
      static const size_t MaximumWords =
         (MaximumByteLength + sizeof(word) - 1) / sizeof(word);

      std::shared_ptr<PrimeOrderCurve> from_id(PrimeOrderCurveId id);

      /// Creates a generic non-optimized version
      //std::shared_ptr<PrimeOrderCurve> from_params(...);

      /*
      Deserialize scalars rejecting out of range
      Scalar from bytes/bits
      Random scalar
      MulByGenerator -> x coordinate -> scalar
      x coordinate -> scalar
      Multiply two scalars
      Square a scalar
      Add two scalars
      Subtract two scalars
      Scalar inversion (constant time)
      Scalar inversion (variable time)
      Pairwise g*x+h*y

      Testing if point is infinity

      Randomize representation of projective point
      Scalar blinding? (Implicit to mul??)
      */

      class Scalar final {
         private:
            std::shared_ptr<const PrimeOrderCurveId> m_curve;
            std::array<word, MaximumWords> m_value;
      };

      class AffinePoint final {
         public:
            std::vector<uint8_t> serialize() const {
               return m_curve->serialize_point(*this);
            }
         private:
            std::shared_ptr<const PrimeOrderCurve> m_curve;
            std::array<word, MaximumWords> m_x;
            std::array<word, MaximumWords> m_y;
      };

      class ProjectivePoint final {
         public:
            AffinePoint to_affine() const {
               return m_curve->to_affine(*this);
            }
         private:
            std::shared_ptr<const PrimeOrderCurve> m_curve;
            std::array<word, MaximumWords> m_x;
            std::array<word, MaximumWords> m_y;
            std::array<word, MaximumWords> m_z;
      };

      virtual ~PrimeOrderCurve() = default;

      virtual std::optional<const PrimeOrderCurveId> curve_id() const = 0;

      virtual ProjectivePoint mul_by_g(const Scalar& scalar) const = 0;

      virtual AffinePoint to_affine(const ProjectivePoint& pt) const = 0;

      virtual std::vector<uint8_t> serialize_point(const AffinePoint& pt) const = 0;

      virtual std::optional<Scalar> deserialize_scalar(std::span<const uint8_t> bytes) const = 0;
};

std::vector<uint8_t> hash_to_curve(PrimeOrderCurveId curve,
                                   std::string_view hash,
                                   bool random_oracle,
                                   std::span<const uint8_t> input,
                                   std::span<const uint8_t> domain_sep);

std::vector<uint8_t> BOTAN_TEST_API mul_by_g(PrimeOrderCurveId curve, std::span<const uint8_t> scalar);

}  // namespace Botan::PCurve

#endif
