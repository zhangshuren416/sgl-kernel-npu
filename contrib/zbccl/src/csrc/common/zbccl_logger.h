/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ZBCCL_LOGGER_H
#define ZBCCL_LOGGER_H

#include <ctime>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <atomic>
#include <mutex>
#include <unistd.h>
#include <sstream>
#include <algorithm>
#include <sys/time.h>
#include <sys/syscall.h>

namespace zbccl {
using ExternalLog = void (*)(int, const char *);

template <typename T>
void log_recursive(std::ostream& os, T&& arg) {
    os << std::forward<T>(arg);
}

template <typename... Args>
void log_all(std::ostream& os, Args&&... args) {
    (os << ... << std::forward<Args>(args));
}

enum LogLevel : int {
    DEBUG_LEVEL = 0,
    INFO_LEVEL,
    WARN_LEVEL,
    ERROR_LEVEL,
    FATAL_LEVEL,
    BUTT_LEVEL  // no use
};

class OutLogger
{
public:
    static OutLogger &Instance()
    {
        static OutLogger gLogger;
        return gLogger;
    }

    static bool ValidateLevel(int level)
    {
        return level >= DEBUG_LEVEL && level < BUTT_LEVEL;
    }

    inline void SetLogLevel(LogLevel level)
    {
        logLevel_ = level;
    }

    inline const LogLevel &GetLogLevel() const
    {
        return logLevel_;
    }

    inline void SetExternalLogFunction(ExternalLog func, bool forceUpdate = false)
    {
        if (logFunc_ == nullptr || forceUpdate) {
            logFunc_ = func;
        }
    }

    inline ExternalLog GetExternalLogFunction() const {
        return logFunc_;
    }

    inline void Log(int level, const std::string &logMsg)
    {
        if (logFunc_ != nullptr) {
            logFunc_(level, logMsg.c_str());
            return;
        }

        struct timeval tv{};
        char strTime[24];

        gettimeofday(&tv, nullptr);
        time_t timeStamp = tv.tv_sec;
        struct tm localTime{};
        auto result = localtime_r(&timeStamp, &localTime);
        if (result == nullptr) {
            return;
        }
        if (strftime(strTime, sizeof strTime, "%Y-%m-%d %H:%M:%S.", result) != 0) {
            const uint8_t TIME_WIDTH = 6U;
            std::cout << strTime << std::setw(TIME_WIDTH) << std::setfill('0') << tv.tv_usec << " "
                      << LogLevelDesc(level) << syscall(SYS_gettid) << logMsg << std::endl;
        } else {
            std::cout << " Invalid time " << LogLevelDesc(level) << syscall(SYS_gettid) << logMsg << std::endl;
        }
        // LCOV_EXCL_STOP
    }

    OutLogger(const OutLogger &) = delete;
    OutLogger(OutLogger &&) = delete;
    OutLogger &operator=(const OutLogger &) = delete;
    OutLogger &operator=(OutLogger &&) = delete;

    ~OutLogger()
    {
        logFunc_ = nullptr;
    }

private:
    OutLogger() = default;

    const char *LogLevelDesc(const int level) const
    {
        const static std::string invalid = "invalid";
        if (UNLIKELY(level < DEBUG_LEVEL || level >= BUTT_LEVEL)) {
            return invalid.c_str();
        }
        return logLevelDesc_[level];
    }

private:
    LogLevel logLevel_ = ERROR_LEVEL;
    ExternalLog logFunc_ = nullptr;

    const char *logLevelDesc_[BUTT_LEVEL] = {"debug", "info", "warn", "error", "fatal"};
};
}  // namespace zbccl

// macro for log
#ifndef UT_ENABLED
#define ZBCCL_LOG_FILENAME_SHORT (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#else
#define ZBCCL_LOG_FILENAME_SHORT (__FILE__)
#endif

#define ZBCCL_LOG_FORMAT ZBCCL_LOG_FILENAME_SHORT << ":" << __LINE__ << " " << __FUNCTION__ << "] "
#define ZBCCL_OUT_LOG(TAG, LEVEL, ARGS)                                             \
    do {                                                                            \
        if (static_cast<int>(LEVEL) < zbccl::OutLogger::Instance().GetLogLevel()) { \
            break;                                                                  \
        }                                                                           \
        std::ostringstream oss;                                                     \
        oss << (TAG) << ZBCCL_LOG_FORMAT << ARGS;                                   \
        zbccl::OutLogger::Instance().Log(static_cast<int>(LEVEL), oss.str());       \
    } while (0)

#define ZBCCL_LOG_DEBUG(ARGS) ZBCCL_OUT_LOG("[ZBCCL ", zbccl::DEBUG_LEVEL, ARGS)
#define ZBCCL_LOG_INFO(ARGS) ZBCCL_OUT_LOG("[ZBCCL ", zbccl::INFO_LEVEL, ARGS)
#define ZBCCL_LOG_WARN(ARGS) ZBCCL_OUT_LOG("[ZBCCL ", zbccl::WARN_LEVEL, ARGS)
#define ZBCCL_LOG_WARN_LIMIT(ARGS) ZBCCL_OUT_LOG_LIMIT("[ZBCCL ", zbccl::WARN_LEVEL, ARGS)
#define ZBCCL_LOG_ERROR(ARGS) ZBCCL_OUT_LOG("[ZBCCL ", zbccl::ERROR_LEVEL, ARGS)

#define ZBCCL_ASSERT_RETURN(ARGS, RET)           \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            ZBCCL_LOG_ERROR("Assert " << #ARGS); \
            return RET;                          \
        }                                        \
    } while (0)

#define ZBCCL_LOG_AND_SET_LAST_ERROR(msg)      \
    do {                                       \
        std::stringstream tmpStr;              \
        tmpStr << msg;                         \
        zbccl::ZBLastError::Set(tmpStr.str()); \
        ZBCCL_LOG_ERROR(tmpStr.str());         \
    } while (0)

#define ZBCCL_VALIDATE_RETURN(ARGS, msg, RET)    \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            ZBCCL_LOG_AND_SET_LAST_ERROR(msg);   \
            return RET;                          \
        }                                        \
    } while (0)

#define ZBCCL_ASSERT_RET_VOID(ARGS)              \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            ZBCCL_LOG_ERROR("Assert " << #ARGS); \
            return;                              \
        }                                        \
    } while (0)

#define ZBCCL_ASSERT_RETURN_NOLOG(ARGS, RET)     \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            return RET;                          \
        }                                        \
    } while (0)

#define ZBCCL_ASSERT(ARGS)                       \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            ZBCCL_LOG_ERROR("Assert " << #ARGS); \
        }                                        \
    } while (0)

#define ZBCCL_CHECK_S(condition, ...)                               \
do {                                                                \
    if (!(condition)) {                                             \
        std::ostringstream oss;                                     \
        oss << "[ZBCCL_" << __FILE__ << ":" << __LINE__ << "] "     \
            << "Check failed: " #condition ". ";                    \
        zbccl::log_all(oss, __VA_ARGS__);                           \
        oss << std::endl;                                           \
        throw std::runtime_error(oss.str());                        \
    }                                                               \
} while (0)

#define ZBCCL_ASSERT_S(condition, ...)                             \
do {                                                               \
    if (!(condition)) {                                            \
        std::ostringstream oss;                                    \
        oss << "[ZBCCL_" << __FILE__ << ":" << __LINE__ << "] "    \
            << "Assertion failed: (" #condition ") ";              \
        zbccl::log_all(oss, __VA_ARGS__);                          \
        oss << std::endl;                                          \
        throw std::runtime_error(oss.str());                       \
    }                                                              \
} while (0)

#endif  // ZBCCL_LOGGER_H
