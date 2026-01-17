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
#ifndef ZBCCL_DEFINES_H
#define ZBCCL_DEFINES_H

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <sstream>

namespace zbccl {
using ZResult = int32_t;

enum ZResultErrorCode : ZResult {
    Z_OK = 0,
    Z_ERROR = -1,
    Z_INVALID_PARAM = -2,
    Z_NEW_OBJ_FAILED = -3,
    Z_DL_OPEN_LIB_FAILED = -4,
    Z_DL_LOAD_SYM_FAILED = -5,
    Z_FILE_NOT_FOUND = -6,
};

#define PATH_MAX_LIMIT 4096L

#define ZBCCL_API __attribute__((visibility("default")))

#define ZBCCL_LOG_ERROR(ARGS)                                                                 \
    do {                                                                                      \
        std::ostringstream oss;                                                               \
        oss << "[ERROR][ZBCCL_ " << __FILE__ << ":" << __LINE__ << "] " << ARGS << std::endl; \
    } while (0)

#define ZBCCL_CHECK(condition, ...)                                        \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::ostringstream oss;                                        \
            oss << "[ERROR][ZBCCL_" << __FILE__ << ":" << __LINE__ << "] " \
                << "Check failed: " #condition ". " << __VA_ARGS__;        \
            throw std::runtime_error(oss.str());                           \
        }                                                                  \
    } while (0)

#define DL_LOAD_SYM(TARGET_FUNC_VAR, TARGET_FUNC_TYPE, FILE_HANDLE, SYMBOL_NAME)                         \
    do {                                                                                                 \
        TARGET_FUNC_VAR = (TARGET_FUNC_TYPE)dlsym(FILE_HANDLE, SYMBOL_NAME);                             \
        if ((TARGET_FUNC_VAR) == nullptr) {                                                              \
            ZBCCL_LOG_ERROR("Failed to call dlsym to load " << (SYMBOL_NAME) << ", error" << dlerror()); \
            dlclose(FILE_HANDLE);                                                                        \
            FILE_HANDLE = nullptr;                                                                       \
            return Z_DL_LOAD_SYM_FAILED;                                                                 \
        }                                                                                                \
    } while (0)

}  // namespace zbccl

#endif
