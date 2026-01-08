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

#include <iostream>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include "acl/acl.h"

namespace zbccl {
using ZResult = int32_t;

enum ZResultErrorCode : ZResult {
    Z_OK = 0,
    Z_ERROR = -1,
    Z_INVALID_PARAM = -2,
    Z_NEW_OBJ_FAILED = -3,
};

enum ZCCLDataType {
    ZCCL_DATA_TYPE_INT8 = 0,
    ZCCL_DATA_TYPE_INT16 = 1,
    ZCCL_DATA_TYPE_INT32 = 2,
    ZCCL_DATA_TYPE_FP16 = 3,
    ZCCL_DATA_TYPE_FP32 = 4,
    ZCCL_DATA_TYPE_INT64 = 5,
    ZCCL_DATA_TYPE_UINT64 = 6,
    ZCCL_DATA_TYPE_UINT8 = 7,
    ZCCL_DATA_TYPE_UINT16 = 8,
    ZCCL_DATA_TYPE_UINT32 = 9,
    ZCCL_DATA_TYPE_FP64 = 10,
    ZCCL_DATA_TYPE_BFP16 = 11,
    ZCCL_DATA_TYPE_RESERVED
};

enum ReduceOp {
    REDUCE_SUM = 0,
    REDUCE_PROD = 1,
    REDUCE_MAX = 2,
    REDUCE_MIN = 3,
    REDUCE_RESERVED = 255
};

#define ZBCCL_API __attribute__((visibility("default")))

// Macro function for unwinding acl errors.
#define CHECK_ACL(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)
}  // namespace zbccl

#endif
