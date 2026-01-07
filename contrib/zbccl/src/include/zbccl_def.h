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
#ifndef ZBCCL_DEF_H_
#define ZBCCL_DEF_H_

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *zbccl_comm_t;

typedef void *aclrtStream;

typedef enum {
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
} zbccl_datatype_t;

typedef enum {
    REDUCE_SUM = 0,
    REDUCE_PROD = 1,
    REDUCE_MAX = 2,
    REDUCE_MIN = 3,
    REDUCE_RESERVED = 255
} zbccl_reduce_op_t;

typedef struct {
    void *address;
    size_t size;
} zbccl_allocator_options;

#ifdef __cplusplus
}
#endif

#endif  // ZBCCL_DEF_H_
