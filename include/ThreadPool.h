#pragma once
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>
#include <thread>
#include <memory>
#include <cassert>
class ThreadPool {
public:
	//构造函数--explicit是为了防止隐形转换
    ThreadPool(int thread_num):is_close(false) {
        //assert(thread_num > 0);
        for(int i = 0; i < thread_num; ++i) {
            std::thread([this](){
                std::unique_lock<std::mutex>ul(this->mx);
                while(true) {
                    if(!this->tasks.empty()) {
                        auto task = std::move(this->tasks.front());
                        this->tasks.pop_front();
                        ul.unlock();
                        task();
                        ul.lock();
                    }
                    else if(this->is_close) break;
                    else this->cv.wait(ul);
                }
            }).detach();
        }
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lg(this->mx);
            this->is_close = true;
        }
        this->cv.notify_all();
    }
    //因为模板函数，所以注意「完美转发」问题
    template<typename T>
    void AddTask(T&& task){
        {
            std::lock_guard<std::mutex> lg(this->mx);
            this->tasks.push_back(std::forward<T>(task));
        }
        this->cv.notify_one(); 
    }

private:

    std::mutex mx;
    std::condition_variable cv;
    std::deque<std::function<void()>> tasks;
    bool is_close;

  //std::shared_ptr<Pool> this;
};
