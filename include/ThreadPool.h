#ifndef THREADPOOL_H_
#define THREADPOOL_H_
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ConcurrentQueue.h"
#include "LockFreeQueue.h"
#include "SafeQueue.h"
class ThreadPool {
  using Task = std::function<void()>;

 private:
  std::vector<std::thread> m_workers_;  // 线程池
  // ConcurrentQueue<Task> m_tasks_;       // 任务队列
  SafeQueue<Task> m_tasks_;         // 任务队列
  std::mutex m_mutex_;              // 线程池互斥锁
  std::condition_variable m_cond_;  // 线程池条件变量
  std::atomic<bool> m_stop_;        // 是否结束线程池中线程
  std::atomic<int> m_threadnum_;    // 线程池中线程数量
 private:
  // 线程池每次执行的函数
  void process_task();

 public:
  ThreadPool(size_t thread_num = 4);
  ~ThreadPool();
  template <typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
};

void ThreadPool::process_task() {
  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(this->m_mutex_);
      // wait直到有task可以执行
      this->m_cond_.wait(
          lock, [this] { return this->m_stop_ || !this->m_tasks_.empty(); });
      if (this->m_stop_ && this->m_tasks_.empty()) return;
      this->m_tasks_.dequeue(task);
    }
    // 执行task
    (task)();
  }
}

ThreadPool::ThreadPool(size_t thread_num) : m_stop_(false) {
  m_threadnum_ = thread_num < 1 ? 1 : thread_num;
  for (size_t i = 0; i < thread_num; ++i) {
    // 初始化执行的函数
    m_workers_.emplace_back([this] { this->process_task(); });
  }
  std::cout << "Threads num: " << m_workers_.size() << std::endl;
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(m_mutex_);
    m_stop_ = true;
  }
  // 唤醒所有线程
  m_cond_.notify_all();
  // 令所有线程退出
  for (std::thread& worker : m_workers_) {
    worker.join();
  }
  std::cout << m_workers_.size() << std::endl;
}

template <typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  // 如果线程池已经终止，则抛出异常
  if (m_stop_.load()) {
    throw std::runtime_error("enqueue on stopped ThreadPool");
  }

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));
  std::future<return_type> res = task->get_future();
  // 添加任务到队列
  {
    std::lock_guard<std::mutex> lock(m_mutex_);
    m_tasks_.enqueue([task]() { (*task)(); });
  }
  // 唤醒一个线程执行
  m_cond_.notify_one();
  return res;
}
#endif /* THREADPOOL_H_ */