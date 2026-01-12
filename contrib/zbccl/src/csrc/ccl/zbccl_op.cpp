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

#include "zbccl_op.h"

using namespace zbccl;

#ifdef __cplusplus
extern "C" {
#endif
ZBCCL_API int32_t zbccl_reduce_scatter(const void *send_buff, void *recv_buff, size_t recv_count,
                                       zbccl_datatype_t data_type, zbccl_reduce_op_t op, int team_id,
                                       aclrtStream stream)
{
    return ZcclReduceScatter(static_cast<uint8_t *>(const_cast<void *>(send_buff)), static_cast<uint8_t *>(recv_buff),
                             recv_count, static_cast<ZCCLDataType>(data_type), team_id, stream,
                             static_cast<uint32_t>(op));
}

ZBCCL_API int32_t zbccl_all_reduce(const void *send_buff, void *recv_buff, size_t count, zbccl_datatype_t data_type,
                                   zbccl_reduce_op_t op, int team_id, aclrtStream stream)
{
    return ZcclAllReduceZeroBuff(static_cast<uint8_t *>(const_cast<void *>(send_buff)),
                                 static_cast<uint8_t *>(recv_buff), count, static_cast<ZCCLDataType>(data_type),
                                 team_id, stream, static_cast<uint32_t>(op));
}

#ifdef __cplusplus
}
#endif
