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
#ifndef ZBCCL_OP_ALL_REDUCE_H
#define ZBCCL_OP_ALL_REDUCE_H

#include "acl/acl.h"
#include "zbccl_defines.h"
#include "zbccl_functions.h"
#include "tiling/platform/platform_ascendc.h"
#include "shmem_api.h"

namespace zbccl {

ZBCCL_API int ZcclAllReduceZeroBuff(uint8_t *inp, uint8_t *out,
    size_t inpNumel, ZCCLDataType dataType, int teamId, aclrtStream stream, uint32_t reduceOp);

}

#endif  // ZBCCL_OP_ALL_REDUCE_H
