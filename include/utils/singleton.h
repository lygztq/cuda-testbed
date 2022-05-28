#ifndef UTILS_SINGLETON_H_
#define UTILS_SINGLETON_H_

namespace utils {

template <typename T>
class ThreadSingleton {
public:
  static T& GetThreadLocal() const {
    static thread_local T obj;
    return obj;
  }
};

template <typename T>
class GlobalSingleton {
public:
  static T& GetThreadLocal() const {
    static T obj;
    return obj;
  }
};

} // namespace utils


#endif  // UTILS_SINGLETON_H_
