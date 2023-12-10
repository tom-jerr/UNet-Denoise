/*
Copyright [2023] <Copyright LZY>
*/
#include "../include/CLLog.h"
namespace neo {
Log* Log::m_log_instance_ = nullptr;
Log::Log(const std::string& path) { m_logFileName_ = path; }

void Log::writeLog(const char* logMsg) {
  FILE* fp = fopen(m_logFileName_.c_str(), "a+");
  if (fp == nullptr) {
    return;
  }
  fprintf(fp, "%s\n", logMsg);
  fclose(fp);
}
}  // namespace neo
