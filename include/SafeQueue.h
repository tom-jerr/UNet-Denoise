#ifndef SAFEm_queue_H_
#define SAFEm_queue_H_
#include <mutex>
#include <queue>
template <typename T>
class SafeQueue {
 private:
  std::queue<T> m_queue_;  // use STL queue to store data
  std::mutex m_mutex_;     // the mutex to synchronise on
 public:
  SafeQueue() = default;
  ~SafeQueue() = default;
  bool enqueue(T t);   // enqueue an object
  bool dequeue(T& t);  // dequeue an object
  bool empty();        // check if queue is empty
  int size();          // return the size of queue
};

template <typename T>
bool SafeQueue<T>::empty() {
  std::lock_guard<std::mutex> lock(m_mutex_);
  return m_queue_.empty();
}

template <typename T>
int SafeQueue<T>::size() {
  std::lock_guard<std::mutex> lock(m_mutex_);
  return m_queue_.size();
}

template <typename T>
bool SafeQueue<T>::enqueue(T t) {
  std::lock_guard<std::mutex> lock(m_mutex_);
  m_queue_.emplace(t);
  return true;
}

template <typename T>
bool SafeQueue<T>::dequeue(T& t) {
  std::lock_guard<std::mutex> lock(m_mutex_);
  if (m_queue_.empty()) {
    return false;
  }
  t = std::move(m_queue_.front());
  m_queue_.pop();
  return true;
}

#endif