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
    Z_INVALID_VALUE = -7,
    Z_INVALID_PTR = -8,
    Z_ERROR_ALLOC = -9,
    Z_NOT_ENOUGH_MEM = -10,
    Z_RT_ERROR = -11,
    Z_LOAD_BOOTSTRAP_LIBRARY_FAILED = -12,
    Z_INIT_BOOTSTRAP_FAILED = -13,
    Z_CANNOT_UNBOOTSTRAP = -14,
    Z_NOT_BOOTSTRAPPED = -15,
    Z_CREATE_COMM_FAILED = -16,
    Z_DL_FUNCTION_UNLOAD = -17,
    Z_CCL_EXEC_FAILED = -18,
    Z_CCL_NOT_EXIST_BY_NAME = -19,
};

#define USE_GITCODE_SHMEM

constexpr uint32_t PATH_MAX_LIMIT = 4096;
constexpr uint32_t RANK_COUNT_MAX_LIMIT = 1024;
constexpr uint32_t DEVICE_COUNT_MAX_LIMIT = 32;
constexpr uint64_t MEMORY_SIZE_CAP = 274877906944; /* 256GB */

#ifndef LIKELY
#define LIKELY(x) (__builtin_expect(!!(x), 1) != 0)
#endif

#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0) != 0)
#endif

#define ZBCCL_LIKELY(expr) LIKELY(expr)
#define ZBCCL_UNLIKELY(expr) UNLIKELY(expr)

#define ZBCCL_API __attribute__((visibility("default")))

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
