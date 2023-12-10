/*
Copyright [2023] <Copyright LZY>
*/
#ifndef INCLUDE_CLLOG_H_
#define INCLUDE_CLLOG_H_
#include <cstdio>
#include <string>

namespace neo {
class Log {
 private:
  explicit Log(const std::string& path);

  std::string m_logFileName_;

 public:
  static Log* m_log_instance_;
  static Log* getLogInstance(const std::string& path) {
    if (m_log_instance_ == nullptr) {
      m_log_instance_ = new Log(path);
    }
    return m_log_instance_;
  }
  void writeLog(const char* logMsg);
  std::string& getLogFileName() { return m_logFileName_; }
};
}  // namespace neo
#endif  // INCLUDE_CLLOG_H_
