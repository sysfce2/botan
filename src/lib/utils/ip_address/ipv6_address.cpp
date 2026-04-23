/*
* (C) 2026 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <botan/ipv6_address.h>

#include <botan/ipv4_address.h>
#include <botan/internal/fmt.h>
#include <botan/internal/loadstor.h>
#include <botan/internal/parsing.h>
#include <bit>

namespace Botan {

IPv6Address::IPv6Address(std::span<const uint8_t, 16> ip) : m_ip{} {
   for(size_t i = 0; i != 16; ++i) {
      m_ip[i] = ip[i];
   }
}

//static
std::optional<IPv6Address> IPv6Address::from_string(std::string_view str) {
   if(auto ipv6 = string_to_ipv6(str)) {
      return IPv6Address(*ipv6);
   } else {
      return {};
   }
}

//static
IPv6Address IPv6Address::netmask(size_t bits) {
   BOTAN_ARG_CHECK(bits <= 128, "IPv6 netmask prefix length must be at most 128");

   const size_t full_bytes = bits / 8;
   const size_t leftover = bits % 8;

   std::array<uint8_t, 16> m{};
   for(size_t i = 0; i != full_bytes; ++i) {
      m[i] = 0xFF;
   }

   if(leftover > 0) {
      m[full_bytes] = static_cast<uint8_t>(0xFF << (8 - leftover));
   }

   return IPv6Address(m);
}

std::string IPv6Address::to_string() const {
   return ipv6_to_string(m_ip);
}

IPv6Address IPv6Address::operator&(const IPv6Address& other) const {
   std::array<uint8_t, 16> masked{};
   for(size_t i = 0; i != 16; ++i) {
      masked[i] = m_ip[i] & other.m_ip[i];
   }
   return IPv6Address(masked);
}

std::optional<size_t> IPv6Address::prefix_length() const {
   // Count leading one bits, stopping at the first byte that isn't fully set.
   size_t leading = 0;
   for(size_t i = 0; i != 16; ++i) {
      const size_t hw = (m_ip[i] == 0xFF) ? 8 : std::countl_one(m_ip[i]);
      leading += hw;
      if(hw != 8) {
         break;
      }
   }

   // Verify this is exactly equal to a netmask of that size
   if(*this != netmask(leading)) {
      return std::nullopt;
   }
   return leading;
}

std::optional<IPv4Address> IPv6Address::as_ipv4() const {
   const uint32_t ip0 = load_be<uint32_t>(m_ip.data(), 0);
   const uint32_t ip1 = load_be<uint32_t>(m_ip.data(), 1);
   const uint32_t ip2 = load_be<uint32_t>(m_ip.data(), 2);
   const uint32_t ip3 = load_be<uint32_t>(m_ip.data(), 3);

   if(ip0 == 0x00000000 && ip1 == 0x00000000 && (ip2 == 0x00000000 || ip2 == 0x0000FFFF)) {
      return IPv4Address(ip3);
   } else {
      return {};
   }
}

IPv6Subnet::IPv6Subnet(IPv6Address address, size_t prefix_length) :
      m_address(address & IPv6Address::netmask(prefix_length)), m_prefix_length(static_cast<uint8_t>(prefix_length)) {
   // IPv6Address::netmask validates prefix_length <= 128, so by this point
   // the static_cast is in range.
}

//static
std::optional<IPv6Subnet> IPv6Subnet::from_address_and_mask(std::span<const uint8_t, 32> addr_and_mask) {
   const auto addr = IPv6Address(addr_and_mask.first<16>());
   const auto mask = IPv6Address(addr_and_mask.last<16>());

   if(const auto plen = mask.prefix_length()) {
      return IPv6Subnet(addr, *plen);
   } else {
      return {};
   }
}

//static
std::optional<IPv6Subnet> IPv6Subnet::from_string(std::string_view str) {
   const auto slash = str.find('/');
   if(slash == std::string_view::npos) {
      return std::nullopt;
   }

   auto addr = IPv6Address::from_string(str.substr(0, slash));
   if(!addr.has_value()) {
      return std::nullopt;
   }

   // Parse the prefix length as a decimal integer in [0, 128].
   const auto plen_str = str.substr(slash + 1);
   if(plen_str.empty() || plen_str.size() > 3) {
      return std::nullopt;
   }
   size_t plen = 0;
   for(const char c : plen_str) {
      if(c < '0' || c > '9') {
         return std::nullopt;
      }
      plen = plen * 10 + static_cast<size_t>(c - '0');
   }
   if(plen > 128) {
      return std::nullopt;
   }

   return IPv6Subnet(*addr, plen);
}

bool IPv6Subnet::contains(const IPv6Address& ip) const {
   return (ip & IPv6Address::netmask(m_prefix_length)) == m_address;
}

std::string IPv6Subnet::to_string() const {
   return fmt("{}/{}", m_address.to_string(), static_cast<size_t>(m_prefix_length));
}

std::vector<uint8_t> IPv6Subnet::serialize() const {
   const auto addr = m_address.address();
   if(is_host()) {
      return std::vector<uint8_t>(addr.begin(), addr.end());
   }
   const auto mask = IPv6Address::netmask(m_prefix_length).address();
   std::vector<uint8_t> out;
   out.reserve(32);
   out.insert(out.end(), addr.begin(), addr.end());
   out.insert(out.end(), mask.begin(), mask.end());
   return out;
}

}  // namespace Botan
