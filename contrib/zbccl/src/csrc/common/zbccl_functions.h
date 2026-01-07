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
#include "zbccl_defines.h"

namespace zbccl {
class Func
{
public:
};

inline size_t GetSizeFromTypeEnum(ZCCLDataType dtype)
{
    switch (dtype) {
        case ZCCLDataType::ZCCL_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_UINT8:
            return sizeof(uint8_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_UINT16:
            return sizeof(uint16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_UINT32:
            return sizeof(uint32_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_UINT64:
            return sizeof(uint64_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32:
            return sizeof(float);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP64:
            return sizeof(double);
        case ZCCLDataType::ZCCL_DATA_TYPE_BFP16:
            return sizeof(int16_t);
        default:
            return 0;
    }
}


}  // namespace zbccl

#endif  // ZBCCL_FUNCTIONS_H
