/*
* Various string utils and parsing functions
* (C) 1999-2007,2013,2014,2015,2018 Jack Lloyd
* (C) 2015 Simon Warta (Kullo GmbH)
* (C) 2017 René Korthaus, Rohde & Schwarz Cybersecurity
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <botan/internal/parsing.h>

#include <botan/exceptn.h>
#include <botan/internal/fmt.h>
#include <botan/internal/loadstor.h>
#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>

namespace Botan {

uint16_t to_uint16(std::string_view str) {
   const uint32_t x = to_u32bit(str);

   if(x != static_cast<uint16_t>(x)) {
      throw Invalid_Argument("Integer value exceeds 16 bit range");
   }

   return static_cast<uint16_t>(x);
}

uint32_t to_u32bit(std::string_view str_view) {
   const std::string str(str_view);

   // std::stoul is not strict enough. Ensure that str is digit only [0-9]*
   for(const char chr : str) {
      if(chr < '0' || chr > '9') {
         throw Invalid_Argument("to_u32bit invalid decimal string '" + str + "'");
      }
   }

   const unsigned long int x = std::stoul(str);

   if constexpr(sizeof(unsigned long int) > 4) {
      // x might be uint64
      if(x > std::numeric_limits<uint32_t>::max()) {
         throw Invalid_Argument("Integer value of " + str + " exceeds 32 bit range");
      }
   }

   return static_cast<uint32_t>(x);
}

/*
* Parse a SCAN-style algorithm name
*/
std::vector<std::string> parse_algorithm_name(std::string_view scan_name) {
   if(scan_name.find('(') == std::string::npos && scan_name.find(')') == std::string::npos) {
      return {std::string(scan_name)};
   }

   std::string name(scan_name);
   std::string substring;
   std::vector<std::string> elems;
   size_t level = 0;

   elems.push_back(name.substr(0, name.find('(')));
   name = name.substr(name.find('('));

   for(auto i = name.begin(); i != name.end(); ++i) {
      const char c = *i;

      if(c == '(') {
         ++level;
      }
      if(c == ')') {
         if(level == 1 && i == name.end() - 1) {
            if(elems.size() == 1) {
               elems.push_back(substring.substr(1));
            } else {
               elems.push_back(substring);
            }
            return elems;
         }

         if(level == 0 || (level == 1 && i != name.end() - 1)) {
            throw Invalid_Algorithm_Name(scan_name);
         }
         --level;
      }

      if(c == ',' && level == 1) {
         if(elems.size() == 1) {
            elems.push_back(substring.substr(1));
         } else {
            elems.push_back(substring);
         }
         substring.clear();
      } else {
         substring += c;
      }
   }

   if(!substring.empty()) {
      throw Invalid_Algorithm_Name(scan_name);
   }

   return elems;
}

std::vector<std::string> split_on(std::string_view str, char delim) {
   std::vector<std::string> elems;
   if(str.empty()) {
      return elems;
   }

   std::string substr;
   for(const char c : str) {
      if(c == delim) {
         if(!substr.empty()) {
            elems.push_back(substr);
         }
         substr.clear();
      } else {
         substr += c;
      }
   }

   if(substr.empty()) {
      throw Invalid_Argument(fmt("Unable to split string '{}", str));
   }
   elems.push_back(substr);

   return elems;
}

/*
* Join a string
*/
std::string string_join(const std::vector<std::string>& strs, char delim) {
   std::ostringstream out;

   for(size_t i = 0; i != strs.size(); ++i) {
      if(i != 0) {
         out << delim;
      }
      out << strs[i];
   }

   return out.str();
}

/*
* Convert a decimal-dotted string to binary IP
*/
std::optional<uint32_t> string_to_ipv4(std::string_view str) {
   // At least 3 dots + 4 1-digit integers
   // At most 3 dots + 4 3-digit integers
   if(str.size() < 3 + 4 * 1 || str.size() > 3 + 4 * 3) {
      return {};
   }

   // the final result
   uint32_t ip = 0;
   // the number of '.' seen so far
   size_t dots = 0;
   // accumulates one quad (range 0-255)
   uint32_t accum = 0;
   // # of digits pushed to accum since last dot
   size_t cur_digits = 0;

   for(const char c : str) {
      if(c == '.') {
         // . without preceding digit is invalid
         if(cur_digits == 0) {
            return {};
         }
         dots += 1;
         // too many dots
         if(dots > 3) {
            return {};
         }

         cur_digits = 0;
         ip = (ip << 8) | accum;
         accum = 0;
      } else if(c >= '0' && c <= '9') {
         const auto d = static_cast<uint8_t>(c - '0');

         // prohibit leading zero in quad (used for octal)
         if(cur_digits > 0 && accum == 0) {
            return {};
         }
         accum = (accum * 10) + d;

         if(accum > 255) {
            return {};
         }

         cur_digits++;
         BOTAN_ASSERT_NOMSG(cur_digits <= 3);
      } else {
         return {};
      }
   }

   // no trailing digits?
   if(cur_digits == 0) {
      return {};
   }

   // insufficient # of dots
   if(dots != 3) {
      return {};
   }

   ip = (ip << 8) | accum;

   return ip;
}

std::optional<std::array<uint8_t, 16>> string_to_ipv6(std::string_view str) {
   if(str.empty()) {
      return {};
   }

   // Parsed hex groups, split by whether they appeared before or after a "::".
   // If no "::" appears, only `pre` is populated and must reach exactly 8 groups.
   std::array<uint16_t, 8> pre{};
   std::array<uint16_t, 8> post{};
   size_t pre_count = 0;
   size_t post_count = 0;
   bool seen_double_colon = false;

   auto hex_value = [](char c) -> std::optional<uint8_t> {
      if(c >= '0' && c <= '9') {
         return c - '0';
      } else if(c >= 'a' && c <= 'f') {
         return 10 + (c - 'a');
      } else if(c >= 'A' && c <= 'F') {
         return 10 + (c - 'A');
      } else {
         return {};
      }
   };

   size_t idx = 0;
   bool expect_group = true;  // set after any separator, cleared after a group

   while(idx < str.size()) {
      if(str[idx] == ':') {
         if(idx + 1 < str.size() && str[idx + 1] == ':') {
            if(seen_double_colon) {
               return {};  // at most one "::"
            }
            seen_double_colon = true;
            idx += 2;
            expect_group = (idx < str.size());
            continue;
         }
         // single ':' separator between groups — only valid after a group
         if(expect_group) {
            return {};
         }
         expect_group = true;
         idx += 1;
         continue;
      }

      // Parse a hex group of 1..4 digits
      uint32_t group = 0;
      size_t hex_chars = 0;
      while(idx < str.size() && hex_chars < 4) {
         const auto digit = hex_value(str[idx]);
         if(digit.has_value() == false) {
            break;
         }
         group = (group << 4) | static_cast<uint32_t>(digit.value());
         idx += 1;
         hex_chars += 1;
      }
      if(hex_chars == 0) {
         return {};
      }
      // If a 5th hex digit follows, the group is oversized.
      if(hex_chars == 4 && idx < str.size() && hex_value(str[idx]).has_value()) {
         return {};
      }

      if(seen_double_colon) {
         if(post_count >= 8) {
            return {};
         }
         post[post_count++] = static_cast<uint16_t>(group);
      } else {
         if(pre_count >= 8) {
            return {};
         }
         pre[pre_count++] = static_cast<uint16_t>(group);
      }
      expect_group = false;
   }

   // Trailing single ':' is invalid
   if(expect_group) {
      return {};
   }

   const size_t total_groups = pre_count + post_count;
   if(seen_double_colon) {
      // "::" has to cover at least one zero group
      if(total_groups > 7) {
         return {};
      }
   } else {
      if(total_groups != 8) {
         return {};
      }
   }

   std::array<uint8_t, 16> out{};
   for(size_t i = 0; i != pre_count; ++i) {
      out[2 * i] = get_byte<0>(pre[i]);
      out[2 * i + 1] = get_byte<1>(pre[i]);
   }
   const size_t gap = 8 - total_groups;
   for(size_t i = 0; i != post_count; ++i) {
      const size_t target = pre_count + gap + i;
      out[2 * target] = get_byte<0>(post[i]);
      out[2 * target + 1] = get_byte<1>(post[i]);
   }
   return out;
}

std::string ipv6_to_string(const std::array<uint8_t, 16>& a) {
   static const char* hex = "0123456789abcdef";

   std::string out;
   out.reserve(39);

   for(size_t i = 0; i != 16; i += 2) {
      if(i != 0) {
         out.push_back(':');
      }
      const uint16_t group = make_uint16(a[i], a[i + 1]);
      bool started = false;
      // Write each nibble omitting leading 0s
      for(int s = 12; s >= 0; s -= 4) {
         const auto nibble = (group >> s) & 0xF;
         if(nibble != 0 || started || s == 0) {
            out.push_back(hex[nibble]);
            started = true;
         }
      }
   }
   return out;
}

/*
* Convert an IP address to decimal-dotted string
*/
std::string ipv4_to_string(uint32_t ip) {
   uint8_t bits[4];
   store_be(ip, bits);

   std::string str;

   for(size_t i = 0; i != 4; ++i) {
      if(i > 0) {
         str += ".";
      }
      str += std::to_string(bits[i]);
   }

   return str;
}

std::string tolower_string(std::string_view str) {
   std::string lower(str);
   for(char& c : lower) {
      const int cu = static_cast<unsigned char>(c);
      if(std::isalpha(cu) != 0) {
         c = static_cast<char>(std::tolower(cu));
      }
   }
   return lower;
}

bool host_wildcard_match(std::string_view issued, std::string_view host) {
   if(host.empty() || issued.empty()) {
      return false;
   }

   // Maximum valid DNS name
   if(host.size() > 253) {
      return false;
   }

   /*
   The wildcard if existing absorbs (host.size() - issued.size() + 1) chars,
   which must be non-negative. So issued cannot possibly exceed host.size() + 1.
   */
   if(issued.size() > host.size() + 1) {
      return false;
   }

   /*
   If there are embedded nulls in your issued name
   Well I feel bad for you son
   */
   if(issued.find('\0') != std::string_view::npos) {
      return false;
   }

   // '*' is not a valid character in DNS names so should not appear on the host side
   if(host.find('*') != std::string_view::npos) {
      return false;
   }

   // Similarly a DNS name can't end in .
   if(host.back() == '.') {
      return false;
   }

   // And a host can't have an empty name component, so reject that
   if(host.find("..") != std::string_view::npos) {
      return false;
   }

   // ASCII-only case-insensitive char equality, avoids locale overhead from tolower
   auto dns_char_eq = [](char a, char b) -> bool {
      if(a == b) {
         return true;
      }
      const auto la = static_cast<unsigned char>(a | 0x20);
      const auto lb = static_cast<unsigned char>(b | 0x20);
      return la == lb && la >= 'a' && la <= 'z';
   };

   auto dns_char_eq_range = [&](std::string_view a, std::string_view b) -> bool {
      if(a.size() != b.size()) {
         return false;
      }
      for(size_t i = 0; i != a.size(); ++i) {
         if(!dns_char_eq(a[i], b[i])) {
            return false;
         }
      }
      return true;
   };

   // Exact match: accept
   if(dns_char_eq_range(issued, host)) {
      return true;
   }

   // First detect offset of wildcard '*' if included
   const size_t first_star = issued.find('*');
   const bool has_wildcard = (first_star != std::string_view::npos);

   // At most one wildcard is allowed
   if(has_wildcard && issued.find('*', first_star + 1) != std::string_view::npos) {
      return false;
   }

   // If no * at all then not a wildcard, and so not a match
   if(!has_wildcard) {
      return false;
   }

   /*
   Now walk through the issued string, making sure every character
   matches. When we come to the (singular) '*', jump forward in the
   hostname by the corresponding amount. We know exactly how much
   space the wildcard takes because it must be exactly `len(host) -
   len(issued) + 1 chars`.

   We also verify that the '*' comes in the leftmost component, and
   doesn't skip over any '.' in the hostname.
   */
   size_t dots_seen = 0;
   size_t host_idx = 0;

   for(size_t i = 0; i != issued.size(); ++i) {
      if(issued[i] == '.') {
         dots_seen += 1;
      }

      if(issued[i] == '*') {
         // Fail: wildcard can only come in leftmost component
         if(dots_seen > 0) {
            return false;
         }

         /*
         Since there is only one * we know the tail of the issued and
         hostname must be an exact match. In this case advance host_idx
         to match.
         */
         const size_t advance = (host.size() - issued.size() + 1);

         if(host_idx + advance > host.size()) {  // shouldn't happen
            return false;
         }

         // Can't be any intervening .s that we would have skipped
         for(size_t k = host_idx; k != host_idx + advance; ++k) {
            if(host[k] == '.') {
               return false;
            }
         }

         host_idx += advance;
      } else {
         if(!dns_char_eq(issued[i], host[host_idx])) {
            return false;
         }

         host_idx += 1;
      }
   }

   // Wildcard issued name must have at least 3 components
   if(dots_seen < 2) {
      return false;
   }

   return true;
}

std::string check_and_canonicalize_dns_name(std::string_view name) {
   if(name.size() > 255) {
      throw Decoding_Error("DNS name exceeds maximum allowed length");
   }

   if(name.empty()) {
      throw Decoding_Error("DNS name cannot be empty");
   }

   if(name.starts_with(".") || name.ends_with(".")) {
      throw Decoding_Error("DNS name cannot start or end with a dot");
   }

   /*
   * Table mapping uppercase to lowercase and only including values for valid DNS names
   * namely A-Z, a-z, 0-9, hyphen, and dot, plus '*' for wildcarding. (RFC 1035)
   */
   // clang-format off
   constexpr uint8_t DNS_CHAR_MAPPING[128] = {
      '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
      '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
      '\0', '\0', '\0', '\0',  '*', '\0', '\0',  '-',  '.', '\0',  '0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',
       '9', '\0', '\0', '\0', '\0', '\0', '\0', '\0',  'a',  'b',  'c',  'd',  'e',  'f',  'g',  'h',  'i',  'j',  'k',
       'l',  'm',  'n',  'o',  'p',  'q',  'r',  's',  't',  'u',  'v',  'w',  'x',  'y',  'z', '\0', '\0', '\0', '\0',
      '\0', '\0',  'a',  'b',  'c',  'd',  'e',  'f',  'g',  'h',  'i',  'j',  'k',  'l',  'm',  'n',  'o',  'p',  'q',
       'r',  's',  't',  'u',  'v',  'w',  'x',  'y',  'z', '\0', '\0', '\0', '\0', '\0',
   };
   // clang-format on

   std::string canon;
   canon.reserve(name.size());

   // RFC 1035: DNS labels must not exceed 63 characters
   size_t current_label_length = 0;

   for(size_t i = 0; i != name.size(); ++i) {
      const char c = name[i];

      if(c == '.') {
         if(i > 0 && name[i - 1] == '.') {
            throw Decoding_Error("DNS name contains sequential period chars");
         }

         if(current_label_length == 0) {
            throw Decoding_Error("DNS name contains empty label");
         }
         current_label_length = 0;  // Reset for next label
      } else {
         current_label_length++;

         if(current_label_length > 63) {  // RFC 1035 Maximum DNS label length
            throw Decoding_Error("DNS name label exceeds maximum length of 63 characters");
         }
      }

      const uint8_t cu = static_cast<uint8_t>(c);
      if(cu >= 128) {
         throw Decoding_Error("DNS name must not contain any extended ASCII code points");
      }
      const uint8_t mapped = DNS_CHAR_MAPPING[cu];
      if(mapped == 0) {
         throw Decoding_Error("DNS name includes invalid character");
      }

      if(mapped == '-') {
         if(i == 0 || (i > 0 && name[i - 1] == '.')) {
            throw Decoding_Error("DNS name has label with leading hyphen");
         } else if(i == name.size() - 1 || (i < name.size() - 1 && name[i + 1] == '.')) {
            throw Decoding_Error("DNS name has label with trailing hyphen");
         }
      }
      canon.push_back(static_cast<char>(mapped));
   }

   if(current_label_length == 0) {
      throw Decoding_Error("DNS name contains empty label");
   }
   return canon;
}

}  // namespace Botan
