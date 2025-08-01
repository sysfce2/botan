/*
* (C) 2019,2021 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <botan/internal/thread_pool.h>

#include <botan/exceptn.h>
#include <botan/internal/os_utils.h>
#include <botan/internal/target_info.h>
#include <algorithm>
#include <thread>

namespace Botan {

namespace {

std::optional<size_t> global_thread_pool_size() {
   std::string var;
   if(OS::read_env_variable(var, "BOTAN_THREAD_POOL_SIZE")) {
      if(var == "none") {
         return std::nullopt;
      }

      // Try to convert to an integer if possible:
      try {
         return std::optional<size_t>(std::stoul(var, nullptr));
      } catch(std::exception&) {}

      // If it was neither a number nor a special value, then ignore the env
   }

   /*
   * On a few platforms, disable the thread pool by default; it is only
   * used if a size is set explicitly in the environment.
   */

#if defined(BOTAN_TARGET_OS_IS_MINGW)
   // MinGW seems to have bugs causing deadlock on application exit.
   // See https://github.com/randombit/botan/issues/2582 for background.
   return std::nullopt;
#elif defined(BOTAN_TARGET_OS_IS_EMSCRIPTEN)
   // Emscripten's threads are reportedly problematic
   // See https://github.com/randombit/botan/issues/4195
   return std::nullopt;
#else
   // Some(0) means choose based on CPU count
   return std::optional<size_t>(0);
#endif
}

}  // namespace

//static
Thread_Pool& Thread_Pool::global_instance() {
   static Thread_Pool g_thread_pool(global_thread_pool_size());
   return g_thread_pool;
}

Thread_Pool::Thread_Pool(std::optional<size_t> opt_pool_size) : m_shutdown(false) {
   // On Linux, it is 16 length max, including terminator
   const std::string tname = "Botan thread";

   if(!opt_pool_size.has_value()) {
      return;
   }

   size_t pool_size = opt_pool_size.value();

   if(pool_size == 0) {
      /*
      * For large machines don't create too many threads, unless
      * explicitly asked to by the caller.
      */
      const size_t cores = OS::get_cpu_available();
      pool_size = std::clamp<size_t>(cores, 2, 16);
   }

   m_workers.resize(pool_size);

   for(size_t i = 0; i != pool_size; ++i) {
      m_workers[i] = std::thread(&Thread_Pool::worker_thread, this);
      OS::set_thread_name(m_workers[i], tname);
   }
}

void Thread_Pool::shutdown() {
   {
      std::unique_lock<std::mutex> lock(m_mutex);

      if(m_shutdown) {
         return;
      }

      m_shutdown = true;

      m_more_tasks.notify_all();
   }

   for(auto&& thread : m_workers) {
      thread.join();
   }
   m_workers.clear();
}

void Thread_Pool::queue_thunk(const std::function<void()>& work) {
   std::unique_lock<std::mutex> lock(m_mutex);

   if(m_shutdown) {
      throw Invalid_State("Cannot add work after thread pool has shut down");
   }

   if(m_workers.empty()) {
      return work();
   }

   m_tasks.push_back(work);
   m_more_tasks.notify_one();
}

void Thread_Pool::worker_thread() {
   for(;;) {
      std::function<void()> task;

      {
         std::unique_lock<std::mutex> lock(m_mutex);
         m_more_tasks.wait(lock, [this] { return m_shutdown || !m_tasks.empty(); });

         if(m_tasks.empty()) {
            if(m_shutdown) {
               return;
            } else {
               continue;
            }
         }

         task = m_tasks.front();
         m_tasks.pop_front();
      }

      task();
   }
}

}  // namespace Botan
