/*
* Certificate Store
* (C) 1999-2010,2013 Jack Lloyd
* (C) 2017 Fabian Weissberg, Rohde & Schwarz Cybersecurity
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <botan/certstor.h>

#include <botan/asn1_time.h>
#include <botan/assert.h>
#include <botan/data_src.h>
#include <botan/pkix_types.h>
#include <botan/internal/filesystem.h>
#include <algorithm>

namespace Botan {

Certificate_Store::~Certificate_Store() = default;

bool Certificate_Store::certificate_known(const X509_Certificate& cert) const {
   return contains(cert);
}

bool Certificate_Store::contains(const X509_Certificate& searching) const {
   for(const auto& cert : find_all_certs(searching.subject_dn(), searching.subject_key_id())) {
      if(cert == searching) {
         return true;
      }
   }

   return false;
}

std::optional<X509_Certificate> Certificate_Store::find_cert(const X509_DN& subject_dn,
                                                             const std::vector<uint8_t>& key_id) const {
   const auto certs = find_all_certs(subject_dn, key_id);

   if(certs.empty()) {
      return std::nullopt;
   }

   // `count` might be greater than 1, but we'll just select the first match
   return certs.front();
}

std::optional<X509_CRL> Certificate_Store::find_crl_for(const X509_Certificate& /*unused*/) const {
   return std::nullopt;
}

void Certificate_Store_In_Memory::add_certificate(const X509_Certificate& cert) {
   const auto tag = cert.tag();
   if(!m_cert_tags.contains(tag)) {
      m_cert_tags.insert(tag);
      const size_t idx = m_certs.size();
      m_certs.push_back(cert);
      m_dn_to_indices[cert.subject_dn()].push_back(idx);
   }
}

std::vector<X509_DN> Certificate_Store_In_Memory::all_subjects() const {
   std::vector<X509_DN> subjects;
   subjects.reserve(m_certs.size());
   for(const auto& cert : m_certs) {
      subjects.push_back(cert.subject_dn());
   }
   return subjects;
}

std::optional<X509_Certificate> Certificate_Store_In_Memory::find_cert(const X509_DN& subject_dn,
                                                                       const std::vector<uint8_t>& key_id) const {
   const auto it = m_dn_to_indices.find(subject_dn);
   if(it == m_dn_to_indices.end()) {
      return std::nullopt;
   }

   for(const size_t idx : it->second) {
      const auto& cert = m_certs[idx];
      BOTAN_ASSERT_NOMSG(cert.subject_dn() == subject_dn);

      if(!key_id.empty()) {
         const std::vector<uint8_t>& skid = cert.subject_key_id();
         if(!skid.empty() && skid != key_id) {  // no match
            continue;
         }
      }

      return cert;
   }

   return std::nullopt;
}

std::vector<X509_Certificate> Certificate_Store_In_Memory::find_all_certs(const X509_DN& subject_dn,
                                                                          const std::vector<uint8_t>& key_id) const {
   std::vector<X509_Certificate> matches;

   const auto it = m_dn_to_indices.find(subject_dn);
   if(it == m_dn_to_indices.end()) {
      return matches;
   }

   for(const size_t idx : it->second) {
      const auto& cert = m_certs[idx];
      BOTAN_ASSERT_NOMSG(cert.subject_dn() == subject_dn);

      if(!key_id.empty()) {
         const std::vector<uint8_t>& skid = cert.subject_key_id();
         if(!skid.empty() && skid != key_id) {  // no match
            continue;
         }
      }

      matches.push_back(cert);
   }

   return matches;
}

std::optional<X509_Certificate> Certificate_Store_In_Memory::find_cert_by_pubkey_sha1(
   const std::vector<uint8_t>& key_hash) const {
   if(key_hash.size() != 20) {
      throw Invalid_Argument("Certificate_Store_In_Memory::find_cert_by_pubkey_sha1 invalid hash");
   }

   for(const auto& cert : m_certs) {
      if(key_hash == cert.subject_public_key_bitstring_sha1()) {
         return cert;
      }
   }

   return std::nullopt;
}

std::optional<X509_Certificate> Certificate_Store_In_Memory::find_cert_by_raw_subject_dn_sha256(
   const std::vector<uint8_t>& subject_hash) const {
   if(subject_hash.size() != 32) {
      throw Invalid_Argument("Certificate_Store_In_Memory::find_cert_by_raw_subject_dn_sha256 invalid hash");
   }

   for(const auto& cert : m_certs) {
      if(subject_hash == cert.raw_subject_dn_sha256()) {
         return cert;
      }
   }

   return std::nullopt;
}

std::optional<X509_Certificate> Certificate_Store_In_Memory::find_cert_by_issuer_dn_and_serial_number(
   const X509_DN& issuer_dn, std::span<const uint8_t> serial_number) const {
   for(const auto& cert : m_certs) {
      if(cert.issuer_dn() == issuer_dn && std::ranges::equal(cert.serial_number(), serial_number)) {
         return cert;
      }
   }

   return std::nullopt;
}

void Certificate_Store_In_Memory::add_crl(const X509_CRL& crl) {
   const X509_DN& crl_issuer = crl.issuer_dn();

   for(auto& c : m_crls) {
      // Found an update of a previously existing one; replace it
      if(c.issuer_dn() == crl_issuer) {
         if(c.this_update() <= crl.this_update()) {
            c = crl;
         }
         return;
      }
   }

   // Totally new CRL, add to the list
   m_crls.push_back(crl);
}

std::optional<X509_CRL> Certificate_Store_In_Memory::find_crl_for(const X509_Certificate& subject) const {
   const std::vector<uint8_t>& key_id = subject.authority_key_id();

   for(const auto& c : m_crls) {
      // Only compare key ids if set in both call and in the CRL
      if(!key_id.empty()) {
         const std::vector<uint8_t>& akid = c.authority_key_id();

         if(!akid.empty() && akid != key_id) {  // no match
            continue;
         }
      }

      if(c.issuer_dn() == subject.issuer_dn()) {
         return c;
      }
   }

   return {};
}

bool Certificate_Store_In_Memory::contains(const X509_Certificate& cert) const {
   return m_cert_tags.contains(cert.tag());
}

Certificate_Store_In_Memory::Certificate_Store_In_Memory(const X509_Certificate& cert) {
   add_certificate(cert);
}

Certificate_Store_In_Memory::Certificate_Store_In_Memory(const X509_Certificate& cert, const X509_CRL& crl) {
   add_certificate(cert);
   add_crl(crl);
}

#if defined(BOTAN_TARGET_OS_HAS_FILESYSTEM)
Certificate_Store_In_Memory::Certificate_Store_In_Memory(std::string_view dir) {
   if(dir.empty()) {
      return;
   }

   std::vector<std::string> maybe_certs = get_files_recursive(dir);

   if(maybe_certs.empty()) {
      maybe_certs.push_back(std::string(dir));
   }

   for(auto&& cert_file : maybe_certs) {
      try {
         DataSource_Stream src(cert_file, true);
         while(!src.end_of_data()) {
            try {
               add_certificate(X509_Certificate(src));
            } catch(std::exception&) {
               // stop searching for other certificate at first exception
               break;
            }
         }
      } catch(std::exception&) {}
   }
}
#endif

}  // namespace Botan
