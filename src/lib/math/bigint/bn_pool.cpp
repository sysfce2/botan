/*
* (C) 2019 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <botan/bigint.h>

namespace Botan {

void BN_Pool::release(size_t cnt)
   {
   BOTAN_ASSERT_NOMSG(cnt <= m_in_use);

   m_in_use -= cnt;

   const size_t free_elems = m_pool.size() - m_in_use;

   if(m_max_cached > 0 && free_elems > m_max_cached)
      {
      m_pool.resize(m_in_use + m_max_cached);
      }
   }

}
