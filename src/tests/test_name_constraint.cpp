/*
* (C) 2015,2016 Kai Michaelis
*     2026 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include "tests.h"

#if defined(BOTAN_HAS_X509_CERTIFICATES)
   #include <botan/ber_dec.h>
   #include <botan/pkix_types.h>
   #include <botan/x509cert.h>
   #include <botan/x509path.h>
   #include <botan/internal/calendar.h>
#endif

namespace Botan_Tests {

namespace {

#if defined(BOTAN_HAS_X509_CERTIFICATES) && defined(BOTAN_HAS_RSA) && defined(BOTAN_HAS_EMSA_PKCS1) && \
   defined(BOTAN_TARGET_OS_HAS_FILESYSTEM)

class Name_Constraint_Tests final : public Test {
   public:
      std::vector<Test::Result> run() override {
         const std::vector<std::tuple<std::string, std::string, std::string, std::string>> test_cases = {
            std::make_tuple("Root_Email_Name_Constraint.crt",
                            "Invalid_Email_Name_Constraint.crt",
                            "",
                            "Certificate does not pass name constraint"),
            std::make_tuple("Root_DN_Name_Constraint.crt",
                            "Invalid_DN_Name_Constraint.crt",
                            "",
                            "Certificate does not pass name constraint"),
            std::make_tuple("Root_DN_Name_Constraint.crt", "Valid_DN_Name_Constraint.crt", "", "Verified"),
            std::make_tuple(
               "Root_DNS_Name_Constraint.crt", "Valid_DNS_Name_Constraint.crt", "aexample.com", "Verified"),
            std::make_tuple("Root_IP_Name_Constraint.crt", "Valid_IP_Name_Constraint.crt", "", "Verified"),
            std::make_tuple("Root_IP_Name_Constraint.crt",
                            "Invalid_IP_Name_Constraint.crt",
                            "",
                            "Certificate does not pass name constraint"),
         };
         std::vector<Test::Result> results;
         const Botan::Path_Validation_Restrictions restrictions(false, 80);

         const std::chrono::system_clock::time_point validation_time =
            Botan::calendar_point(2016, 10, 21, 4, 20, 0).to_std_timepoint();

         for(const auto& t : test_cases) {
            const Botan::X509_Certificate root(Test::data_file("x509/name_constraint/" + std::get<0>(t)));
            const Botan::X509_Certificate sub(Test::data_file("x509/name_constraint/" + std::get<1>(t)));
            Botan::Certificate_Store_In_Memory trusted;
            Test::Result result("X509v3 Name Constraints: " + std::get<1>(t));

            trusted.add_certificate(root);
            Botan::Path_Validation_Result path_result = Botan::x509_path_validate(
               sub, restrictions, trusted, std::get<2>(t), Botan::Usage_Type::TLS_SERVER_AUTH, validation_time);

            if(path_result.successful_validation() && path_result.trust_root() != root) {
               path_result = Botan::Path_Validation_Result(Botan::Certificate_Status_Code::CANNOT_ESTABLISH_TRUST);
            }

            result.test_str_eq("validation result", path_result.result_string(), std::get<3>(t));
            results.emplace_back(result);
         }

         return results;
      }
};

BOTAN_REGISTER_TEST("x509", "x509_path_name_constraint", Name_Constraint_Tests);

// Verify that DNS constraints are case-insensitive also when falling back to the CN
class Name_Constraint_Excluded_CN_Case_Test final : public Test {
   public:
      std::vector<Test::Result> run() override {
         Test::Result result("X509v3 Name Constraints: excluded DNS with mixed-case CN and no SAN");

         const Botan::X509_Certificate root(
            Test::data_file("x509/name_constraint/Root_DNS_Excluded_Mixed_Case_CN.crt"));
         const Botan::X509_Certificate leaf(
            Test::data_file("x509/name_constraint/Invalid_DNS_Excluded_Mixed_Case_CN.crt"));

         Botan::Certificate_Store_In_Memory trusted;
         trusted.add_certificate(root);

         const Botan::Path_Validation_Restrictions restrictions(false, 80);
         const auto validation_time = Botan::calendar_point(2026, 6, 1, 0, 0, 0).to_std_timepoint();

         const auto path_result = Botan::x509_path_validate(
            leaf, restrictions, trusted, "" /* hostname */, Botan::Usage_Type::UNSPECIFIED, validation_time);

         result.test_str_eq(
            "validation result", path_result.result_string(), "Certificate does not pass name constraint");

         return {result};
      }
};

BOTAN_REGISTER_TEST("x509", "x509_name_constraint_excluded_cn_case", Name_Constraint_Excluded_CN_Case_Test);

class Name_Constraint_IPv6_Chain_Tests final : public Test {
   public:
      std::vector<Test::Result> run() override {
         struct Case {
               std::string label;
               std::string dir;
               std::vector<std::string> intermediates;
               std::string leaf;
               bool accept;
         };

         const std::vector<Case> cases = {
            // IPv6 permittedSubtree 2001:db8::/32
            {"IPv6 permit: SAN inside subtree", "permitted", {}, "leaf_valid.pem", true},
            {"IPv6 permit: SAN outside subtree", "permitted", {}, "leaf_invalid.pem", false},

            // IPv6 excludedSubtree 2001:db8::/32
            {"IPv6 exclude: SAN outside subtree", "excluded", {}, "leaf_valid.pem", true},
            {"IPv6 exclude: SAN inside subtree", "excluded", {}, "leaf_invalid.pem", false},

            // Root permits only IPv4 10.0.0.0/8, so an IPv6 SAN must be rejected
            // because iPAddress is a single GeneralName form (RFC 5280 4.2.1.10).
            {"IPv4-only permit: IPv4 SAN", "cross_v4only", {}, "leaf_valid.pem", true},
            {"IPv4-only permit: IPv6 SAN", "cross_v4only", {}, "leaf_invalid.pem", false},

            // Similar to previous - root permits only IPv6 2001:db8::/32 so IPv4 must be rejected
            {"IPv6-only permit: IPv6 SAN", "cross_v6only", {}, "leaf_valid.pem", true},
            {"IPv6-only permit: IPv4 SAN", "cross_v6only", {}, "leaf_invalid.pem", false},

            // Constraints across multiple issuers
            // - root permits {10/8, 2001:db8::/32}
            // - intermediate narrows permits to {10.1/16, 2001:db8:cafe::/48} and excludes
            //   {10.1.99/24, 2001:db8:cafe:bad::/64}.
            //
            // Every leaf has one IPv4 and one IPv6 SAN
            {"Mixed v4+v6: all SANs in range", "mixed_multi", {"int.pem"}, "leaf_valid.pem", true},
            {"Mixed v4+v6: IPv4 outside int permit", "mixed_multi", {"int.pem"}, "leaf_invalid_int_v4.pem", false},
            {"Mixed v4+v6: IPv6 outside int permit", "mixed_multi", {"int.pem"}, "leaf_invalid_int_v6.pem", false},
            {"Mixed v4+v6: IPv4 hits int exclude", "mixed_multi", {"int.pem"}, "leaf_invalid_excl_v4.pem", false},
            {"Mixed v4+v6: IPv6 hits int exclude", "mixed_multi", {"int.pem"}, "leaf_invalid_excl_v6.pem", false},
            {"Mixed v4+v6: IPv4 outside root permit", "mixed_multi", {"int.pem"}, "leaf_invalid_root_v4.pem", false},
            {"Mixed v4+v6: IPv6 outside root permit", "mixed_multi", {"int.pem"}, "leaf_invalid_root_v6.pem", false},

            // Here the root excludes IPv4 10.0.0.0/8 and the leaf certs have IPv6 SAN; the
            // invalid leaf has a IPv4-mapped IPv6 address matching 10.0.0.0/8 while the valid leaf
            // has some other IPv6 address which is not excluded
            {"v4 exclude: mapped-v6 SAN inside v4 excl", "v4_exclude_mapped", {}, "leaf_invalid.pem", false},
            {"v4 exclude: mapped-v6 SAN outside v4 excl", "v4_exclude_mapped", {}, "leaf_valid.pem", true},

            // Here the root permits only IPv4 10.0.0.0/8. The invalid leaf has an IPv6 address in the SAN
            // which should be rejected as not being in the range. The 'valid' leaf is a questionable case: it
            // has an IPv6 SAN which is an IPv4-mapped IPv6 address inside 10.0.0.0/8. Arguably it really is
            // valid; RFC 5280 is silent on the issue. But lacking a clear consensus, it is rejected for now.
            {"v4 permit: mapped-v6 SAN inside v4 permit", "v4_permit_mapped", {}, "leaf_valid.pem", false},
            {"v4 permit: mapped-v6 SAN outside v4 permit", "v4_permit_mapped", {}, "leaf_invalid.pem", false},
         };

         const Botan::Path_Validation_Restrictions restrictions(false, 80);

         const auto validation_time = Botan::calendar_point(2027, 4, 22, 20, 0, 0).to_std_timepoint();

         std::vector<Test::Result> results;
         for(const auto& c : cases) {
            const std::string base = "x509/name_constraint_ipv6/" + c.dir + "/";
            const Botan::X509_Certificate root(Test::data_file(base + "root.pem"));
            const Botan::X509_Certificate leaf(Test::data_file(base + c.leaf));

            std::vector<Botan::X509_Certificate> chain{leaf};
            for(const auto& intermediate_file : c.intermediates) {
               chain.emplace_back(Test::data_file(base + intermediate_file));
            }

            Botan::Certificate_Store_In_Memory trusted;
            trusted.add_certificate(root);

            const auto pv = Botan::x509_path_validate(
               chain, restrictions, trusted, "" /* hostname */, Botan::Usage_Type::UNSPECIFIED, validation_time);

            Test::Result result("X509v3 Name Constraints (IPv6 chains): " + c.label);

            const std::string expected = c.accept ? "Verified" : "Certificate does not pass name constraint";
            result.test_str_eq("path validation result", pv.result_string(), expected);
            results.emplace_back(std::move(result));
         }

         return results;
      }
};

BOTAN_REGISTER_TEST("x509", "x509_name_constraint_ipv6_chains", Name_Constraint_IPv6_Chain_Tests);

/*
* Validate that GeneralName iPAddress decoding rejects masks that are not a
* contiguous CIDR prefix. Drives the decoder with hand-rolled BER for a
* single [7] IMPLICIT OCTET STRING carrying {net || mask}.
*/
class Name_Constraint_IP_Mask_Tests final : public Text_Based_Test {
   public:
      Name_Constraint_IP_Mask_Tests() : Text_Based_Test("x509/general_name_ip.vec", "Address,Netmask") {}

      Test::Result run_one_test(const std::string& header, const VarMap& vars) override {
         Test::Result result("GeneralName iPAddress mask validation");

         const auto address = vars.get_req_bin("Address");
         const auto netmask = vars.get_req_bin("Netmask");

         const auto der = encode_address(address, netmask);

         Botan::BER_Decoder decoder(der, Botan::BER_Decoder::Limits::DER());
         Botan::GeneralName gn;

         if(header == "Valid") {
            try {
               gn.decode_from(decoder);
               result.test_success("Accepted valid GeneralName IP encoding");
            } catch(Botan::Decoding_Error&) {
               result.test_failure("Rejected valid GeneralName IP encoding");
            }
         } else {
            try {
               gn.decode_from(decoder);
               result.test_failure("Accepted invalid GeneralName IP encoding");
            } catch(Botan::Decoding_Error&) {
               result.test_success("Rejected invalid GeneralName IP encoding");
            }
         }

         return result;
      }

   private:
      static std::vector<uint8_t> encode_address(std::span<const uint8_t> address, std::span<const uint8_t> netmask) {
         std::vector<uint8_t> der;
         // [7] IMPLICIT OCTET STRING, primitive, context-specific.
         der.push_back(0x87);
         // Short for length is sufficient here
         der.push_back(static_cast<uint8_t>(address.size() + netmask.size()));
         der.insert(der.end(), address.begin(), address.end());
         der.insert(der.end(), netmask.begin(), netmask.end());
         return der;
      }
};

BOTAN_REGISTER_TEST("x509", "x509_name_constraint_ip_mask", Name_Constraint_IP_Mask_Tests);

#endif

}  // namespace

}  // namespace Botan_Tests
