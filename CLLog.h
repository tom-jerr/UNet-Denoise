#ifndef INCLUDE_CLLOG_H_
#define INCLUDE_CLLOG_H_
#include <string>
#include <stdio.h>
class Log {
  private:
    Log(const std::string& path);         // 构造函数私有化
  private:
    std::string m_logFileName_;           // 日志文件的路径
  public:
    static Log* m_logInstance;            // 单例模式的对象
    static Log* getLogInstance(const std::string& path) {   // 获取单例模式的对象
      if (m_logInstance == nullptr) {
        m_logInstance = new Log(path);
      }
      return m_logInstance;
    }
    void writeLog(const char* logMsg);    // 写日志
    std::string& getLogFileName() {       // 获取日志文件的路径
	  return m_logFileName_;
	}
};

Log* Log::m_logInstance = nullptr;
Log::Log(const std::string& path) {
  m_logFileName_ = path;
}

void Log::writeLog(const char* logMsg) {
  FILE* fp = fopen(m_logFileName_.c_str(), "a+");
  if (fp == nullptr) {
    return;
  }
  fprintf(fp, "%s\n", logMsg);
  fclose(fp);
}

#endif  // INCLUDE_CLLOG_H_
