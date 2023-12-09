#ifndef CLTHREADPOOL_H_
#define CLTHREADPOOL_H_
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>
#include <thread>
#include <memory>
#include <cassert>
#include "CLDenoise.h"

class ThreadPool {
public:
	//构造函数
    ThreadPool(int thread_num): m_close_(false) {
        for (int i = 0; i < thread_num; ++i) {
            m_workers_.emplace_back(
                [this] {
                    for (;;) {
						DenoiseOP* task = nullptr;
                        {
							std::unique_lock<std::mutex> ul(this->m_mutex_);
                            this->m_cond_.wait(ul, [this] {
								return this->m_close_ || !this->m_tasks_.empty();
							});
                            if (this->m_close_ && this->m_tasks_.empty()) {
								return;
							}
							task = this->m_tasks_.front();
							this->m_tasks_.pop_front();
						}
						assert(task);
						task->DenoiseUML();
						delete task;
					}
				}
			);
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lg(this->m_mutex_);
            this->m_close_ = true;
        }
        this->m_cond_.notify_all();
        for (auto& worker : this->m_workers_) {
			worker.join();
		}
    }
    //因为模板函数，所以注意「完美转发」问题
    void AddTask(DenoiseOP* task){
        {
            std::lock_guard<std::mutex> lg(this->m_mutex_);
            this->m_tasks_.push_back(task);
            std::cout << "add task" << std::endl;
        }
        std::cout << "notify one" << std::endl;
        this->m_cond_.notify_one(); 
    }

private:

    std::mutex m_mutex_;
    std::condition_variable m_cond_;
    std::deque<DenoiseOP*> m_tasks_;
    std::vector<std::thread> m_workers_;
    bool m_close_;

  //std::shared_ptr<Pool> this;
};
#endif /* CLTHREADPOOL_H_ */