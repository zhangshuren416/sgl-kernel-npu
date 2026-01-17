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
    ZBCCL_DATA_TYPE_INT8 = 0,    /**< int8 */
    ZBCCL_DATA_TYPE_INT16 = 1,   /**< int16 */
    ZBCCL_DATA_TYPE_INT32 = 2,   /**< int32 */
    ZBCCL_DATA_TYPE_FP16 = 3,    /**< fp16 */
    ZBCCL_DATA_TYPE_FP32 = 4,    /**< fp32 */
    ZBCCL_DATA_TYPE_INT64 = 5,   /**< int64 */
    ZBCCL_DATA_TYPE_UINT64 = 6,  /**< uint64 */
    ZBCCL_DATA_TYPE_UINT8 = 7,   /**< uint8 */
    ZBCCL_DATA_TYPE_UINT16 = 8,  /**< uint16 */
    ZBCCL_DATA_TYPE_UINT32 = 9,  /**< uint32 */
    ZBCCL_DATA_TYPE_FP64 = 10,   /**< fp64 */
    ZBCCL_DATA_TYPE_BFP16 = 11,  /**< bfp16 */
    ZBCCL_DATA_TYPE_RESERVED     /**< reserved */
} zbccl_datatype_t; // reference to HcclDataType

typedef enum {
    ZBCCL_REDUCE_SUM = 0,    /**< sum */
    ZBCCL_REDUCE_PROD = 1,   /**< prod */
    ZBCCL_REDUCE_MAX = 2,    /**< max */
    ZBCCL_REDUCE_MIN = 3,    /**< min */
    ZBCCL_REDUCE_RESERVED    /**< reserved */
} zbccl_reduce_op_t; // reference to HcclReduceOp

typedef struct {
    void *address;
    size_t size;
} zbccl_allocator_options;

#ifdef __cplusplus
}
#endif

#endif  // ZBCCL_DEF_H_
