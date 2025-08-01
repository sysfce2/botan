/*
* PEM Encoding/Decoding
* (C) 1999-2007 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <botan/pem.h>

#include <botan/base64.h>
#include <botan/data_src.h>
#include <botan/exceptn.h>
#include <botan/internal/fmt.h>

namespace Botan::PEM_Code {

namespace {

std::string linewrap(size_t width, std::string_view in) {
   std::string out;
   for(size_t i = 0; i != in.size(); ++i) {
      if(i > 0 && i % width == 0) {
         out.push_back('\n');
      }
      out.push_back(in[i]);
   }
   if(!out.empty() && out[out.size() - 1] != '\n') {
      out.push_back('\n');
   }

   return out;
}

}  // namespace

/*
* PEM encode BER/DER-encoded objects
*/
std::string encode(const uint8_t der[], size_t length, std::string_view label, size_t width) {
   const std::string PEM_HEADER = fmt("-----BEGIN {}-----\n", label);
   const std::string PEM_TRAILER = fmt("-----END {}-----\n", label);

   return (PEM_HEADER + linewrap(width, base64_encode(der, length)) + PEM_TRAILER);
}

/*
* Decode PEM down to raw BER/DER
*/
secure_vector<uint8_t> decode_check_label(DataSource& source, std::string_view label_want) {
   std::string label_got;
   secure_vector<uint8_t> ber = decode(source, label_got);
   if(label_got != label_want) {
      throw Decoding_Error(fmt("PEM: Label mismatch, wanted '{}' got '{}'", label_want, label_got));
   }

   return ber;
}

/*
* Decode PEM down to raw BER/DER
*/
secure_vector<uint8_t> decode(DataSource& source, std::string& label) {
   const size_t RANDOM_CHAR_LIMIT = 8;

   label.clear();

   const std::string PEM_HEADER1 = "-----BEGIN ";
   const std::string PEM_HEADER2 = "-----";
   size_t position = 0;

   while(position != PEM_HEADER1.length()) {
      auto b = source.read_byte();

      if(!b) {
         throw Decoding_Error("PEM: No PEM header found");
      }
      if(static_cast<char>(*b) == PEM_HEADER1[position]) {
         ++position;
      } else if(position >= RANDOM_CHAR_LIMIT) {
         throw Decoding_Error("PEM: Malformed PEM header");
      } else {
         position = 0;
      }
   }
   position = 0;
   while(position != PEM_HEADER2.length()) {
      auto b = source.read_byte();

      if(!b) {
         throw Decoding_Error("PEM: No PEM header found");
      }
      if(static_cast<char>(*b) == PEM_HEADER2[position]) {
         ++position;
      } else if(position > 0) {
         throw Decoding_Error("PEM: Malformed PEM header");
      }

      if(position == 0) {
         label += static_cast<char>(*b);
      }
   }

   std::vector<char> b64;

   const std::string PEM_TRAILER = fmt("-----END {}-----", label);
   position = 0;
   while(position != PEM_TRAILER.length()) {
      auto b = source.read_byte();

      if(!b) {
         throw Decoding_Error("PEM: No PEM trailer found");
      }
      if(static_cast<char>(*b) == PEM_TRAILER[position]) {
         ++position;
      } else if(position > 0) {
         throw Decoding_Error("PEM: Malformed PEM trailer");
      }

      if(position == 0) {
         b64.push_back(*b);
      }
   }

   return base64_decode(b64.data(), b64.size());
}

secure_vector<uint8_t> decode_check_label(std::string_view pem, std::string_view label_want) {
   DataSource_Memory src(pem);
   return decode_check_label(src, label_want);
}

secure_vector<uint8_t> decode(std::string_view pem, std::string& label) {
   DataSource_Memory src(pem);
   return decode(src, label);
}

/*
* Search for a PEM signature
*/
bool matches(DataSource& source, std::string_view extra, size_t search_range) {
   const std::string PEM_HEADER = fmt("-----BEGIN {}", extra);

   secure_vector<uint8_t> search_buf(search_range);
   const size_t got = source.peek(search_buf.data(), search_buf.size(), 0);

   if(got < PEM_HEADER.length()) {
      return false;
   }

   size_t index = 0;

   for(size_t j = 0; j != got; ++j) {
      if(static_cast<char>(search_buf[j]) == PEM_HEADER[index]) {
         ++index;
      } else {
         index = 0;
      }

      if(index == PEM_HEADER.size()) {
         return true;
      }
   }

   return false;
}

}  // namespace Botan::PEM_Code
