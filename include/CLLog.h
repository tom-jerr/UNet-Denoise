#ifndef CLLOG_H_
#define CLLOG_H_
#include <string>
#include <stdio.h>
class Log {
  private:
    Log(const std::string& path);
  private:
    std::string m_logFileName_;
  public:
    static Log* m_logInstance;
    static Log* getLogInstance(const std::string& path) {
      if (m_logInstance == nullptr) {
        m_logInstance = new Log(path);
      }
      return m_logInstance;
    }
    void writeLog(const char* logMsg);
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

#endif /* CLLOG_H_ */