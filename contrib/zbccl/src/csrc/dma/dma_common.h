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
#ifndef DMA_COMMON_H
#define DMA_COMMON_H

#include <mutex>
#include <atomic>

struct OptionsManager {
    static bool IsHcclZeroCopyEnable;
    static bool CheckForceUncached;
};

std::string formatErrorCode(int32_t errorCode);

#define PTA_ERROR_MOCK(err_code) formatErrorCode((int32_t)err_code)
#define OPS_ERROR_MOCK(err_code) formatErrorCode((int32_t)err_code)

#define NPU_CHECK_ERROR_MOCK(err_code, ...)                                  \
    do {                                                                     \
        int error_code = err_code;                                           \
        if ((error_code) != ACL_ERROR_NONE) {                                \
            std::ostringstream oss;                                          \
            oss << " NPU function error: [ShmemAllocator Currently do not support detail error log]" << std::endl; \
            std::string err_msg = oss.str();                                 \
            ASCEND_LOGE("%s", err_msg.c_str());                              \
        }                                                                    \
    } while (0)

#define NPU_CHECK_WARN_MOCK(err_code, ...)                                   \
    do {                                                                     \
        int error_code = err_code;                                           \
        if ((error_code) != ACL_ERROR_NONE) {                                \
            std::ostringstream oss;                                          \
            oss << " NPU function warning: [ShmemAllocator Currently do not support detail warning log]" << std::endl; \
            std::string err_msg = oss.str();                                 \
            ASCEND_LOGW("%s", err_msg.c_str());                              \
        }                                                                    \
    } while (0)

const int32_t ACL_SYNC_TIMEOUT = 3600 * 1000;  // ms

#endif  // DMA_COMMON_H
