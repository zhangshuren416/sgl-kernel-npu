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

#define ZBCCL_MAX_RANKS 1024

typedef void *zbccl_comm_t;

typedef void *aclrtStream;

typedef enum {
    ZBCCL_ASCEND_NPU = 0,
} zbccl_backend_t;

typedef enum {
    ZBCCL_DATA_TYPE_INT8 = 0,   /**< int8 */
    ZBCCL_DATA_TYPE_INT16 = 1,  /**< int16 */
    ZBCCL_DATA_TYPE_INT32 = 2,  /**< int32 */
    ZBCCL_DATA_TYPE_FP16 = 3,   /**< fp16 */
    ZBCCL_DATA_TYPE_FP32 = 4,   /**< fp32 */
    ZBCCL_DATA_TYPE_INT64 = 5,  /**< int64 */
    ZBCCL_DATA_TYPE_UINT64 = 6, /**< uint64 */
    ZBCCL_DATA_TYPE_UINT8 = 7,  /**< uint8 */
    ZBCCL_DATA_TYPE_UINT16 = 8, /**< uint16 */
    ZBCCL_DATA_TYPE_UINT32 = 9, /**< uint32 */
    ZBCCL_DATA_TYPE_FP64 = 10,  /**< fp64 */
    ZBCCL_DATA_TYPE_BFP16 = 11, /**< bfp16 */
    ZBCCL_DATA_TYPE_RESERVED    /**< reserved */
} zbccl_datatype_t;             // reference to HcclDataType

typedef enum {
    ZBCCL_REDUCE_SUM = 0,  /**< sum */
    ZBCCL_REDUCE_PROD = 1, /**< prod */
    ZBCCL_REDUCE_MAX = 2,  /**< max */
    ZBCCL_REDUCE_MIN = 3,  /**< min */
    ZBCCL_REDUCE_RESERVED  /**< reserved */
} zbccl_reduce_op_t;       // reference to HcclReduceOp

typedef enum {
    BOOT_BY_MEMFABRIC = 0,
    BOOT_BY_ACLSHMEM,

    BOOT_BY_BUTT
} zbccl_bootstrap_type_t;

typedef struct {
    uint32_t flags;                /* optional flags*/
    zbccl_bootstrap_type_t btType; /* bootstrap type */
    char *ipPort;                  /* tcp://127.0.0.1:9897*/
    uint16_t worldSize;            /* how many rank in total */
    uint16_t rankId;               /* my rank id in the world */
    uint16_t deviceId;             /* device id */
    uint16_t startConfigServer;    /* if start config store server, 1 means start, 0 means not start */
    uint64_t deviceMemorySize;     /* memory size can be allocated */
    uint32_t dataOperationType;    /* data operation type */
} zbccl_bootstrap_options_t;

typedef struct {
    void *deviceGva;                    /* gva of the world */
    void *myDeviceGva;                  /* gva of this rank */
    uint64_t allocatedDeviceMemorySize; /* actually allocated memory size */
} zbccl_bootstrap_output_t;

typedef struct {
    void *gva;     /* gva of the world */
    void *myGva;   /* gva of this rank */
    uint64_t size; /* device memory size */
} zbccl_allocator_options;

typedef struct {
    zbccl_backend_t backendType; /* backend type */
    uint32_t flags;              /* optional flags */
    uint16_t isWorldGroup;       /* if this is the world group, 1 means true, 0 means false */
    uint16_t groupSize;          /* how many rank in total */
    uint16_t groupRankId;        /* my rank id in the world */
    uint16_t isolateOpMeta;      /* if isolate the operator meta data */
    void *deviceGva;             /* gva of the world */
    void *myDeviceGva;           /* gva of this rank */
} zbccl_ccl_options_t;

#ifdef __cplusplus
}
#endif

#endif  // ZBCCL_DEF_H_
