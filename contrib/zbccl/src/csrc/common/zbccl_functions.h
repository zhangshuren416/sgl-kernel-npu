/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * ZBCCL is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef ZBCCL_FUNCTIONS_H
#define ZBCCL_FUNCTIONS_H

#include <string>
#include <limits.h>
#include <unistd.h>
#include <sys/param.h>

#include "zbccl_defines.h"

namespace zbccl {
class Func
{
public:
    /**
     * @brief Get real path with dir and file path
     *
     * @param libDirPath   [in] library path
     * @param libName      [in] library name
     * @param realPath     [out] realpath
     * @return true if get real path successfully
     */
    static bool LibraryRealPath(const std::string &libDirPath, const std::string &libName, std::string &realPath);

    /**
     * @brief Get real path for secure consideration
     *
     * @param path         [in/out] source path and output path
     *
     * @return true if get real path successfully
     */
    static bool Realpath(std::string &path);

    /**
     * @brief Get max length of file path
     */
    static constexpr size_t GetSafePathMax();

    /**
     * @brief Get environment variable value
     *
     * @param name                [in] environment variable name
     * @param defaultValue        [in] default value if environment variable is not set or invalid
     * @return the value of the environment variable if set and valid, otherwise the default value
     */
    template <typename T>
    static T GetEnv(const char* name, T defaultValue = T{});
};

inline bool Func::LibraryRealPath(const std::string &libDirPath, const std::string &libName, std::string &realPath)
{
    std::string tmpFullPath = libDirPath;
    if (!Realpath(tmpFullPath)) {
        return false;
    }

    if (tmpFullPath.back() != '/') {
        tmpFullPath.push_back('/');
    }

    tmpFullPath.append(libName);
    auto ret = ::access(tmpFullPath.c_str(), F_OK);
    if (ret != 0) {
        return false;
    }

    realPath = tmpFullPath;
    return true;
}

inline bool Func::Realpath(std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX_LIMIT) {
        return false;
    }

    /* It will allocate memory to store path */
    char *tmp = new (std::nothrow) char[GetSafePathMax() + 1];
    if (tmp == nullptr) {
        return false;
    }

    char *realPath = realpath(path.c_str(), tmp);
    if (realPath == nullptr) {
        delete[] tmp;
        return false;
    }

    path = realPath;
    realPath = nullptr;
    delete[] tmp;
    return true;
}

inline constexpr size_t Func::GetSafePathMax()
{
#ifdef PATH_MAX
    return (PATH_MAX < PATH_MAX_LIMIT) ? PATH_MAX : PATH_MAX_LIMIT;
#else
    return PATH_MAX_LIMIT;
#endif
}

template <typename T>
inline T Func::GetEnv(const char* name, T defaultValue)
{
    const char* envValue = std::getenv(name);
    if (envValue == nullptr) {
        return defaultValue;
    }

    std::string envStr(envValue);
    T result;
    const char* end;
    try {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            result = static_cast<T>(std::stod(envStr));
        } else if constexpr (std::is_integral_v<T>) {
            result = static_cast<T>(std::stoi(envStr));
        } else {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                            std::is_integral_v<T>, "Unsupported type for GetEnv");
        }
        return result;
    }
    catch (const std::invalid_argument& ia) {
        ZBCCL_OP_LOGE("", "Invalid argument when parsing %s: %s", name, ia.what());
    }
    catch (const std::out_of_range& oor) {
        ZBCCL_OP_LOGE("", "Out of range when parsing %s: %s", name, oor.what());
    }
    catch (...) {
        ZBCCL_OP_LOGE("", "Unexpected error when parsing %s", name);
    }
    return defaultValue;
}


}  // namespace zbccl

#endif  // ZBCCL_FUNCTIONS_H
