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
#include "zbccl_communicator_default.h"

namespace zbccl {
namespace ccl {
ZBCCLCommDefault::ZBCCLCommDefault(const ZBCommOptions &options, bool isWorldGroup, const ZBCCLCommPtr &worldGroup)
    : ZBCCLComm(options, isWorldGroup, worldGroup)
{}

ZResult ZBCCLCommDefault::Initialize() noexcept
{
    // TODO
    return Z_OK;
}

void ZBCCLCommDefault::UnInitialize() noexcept
{
    // TODO
}

int32_t ZBCCLCommDefault::AllReduce(const void *send_buff, void *recv_buff, size_t count, zbccl_datatype_t data_type,
                                    zbccl_reduce_op_t op) noexcept
{
    // TODO
    return Z_OK;
}

int32_t ZBCCLCommDefault::ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count,
                                        zbccl_datatype_t data_type, zbccl_reduce_op_t op) noexcept
{
    // TODO
    return Z_OK;
}

int32_t ZBCCLCommDefault::AllGather(const void *send_buff, void *recv_buff, size_t send_count,
                                    zbccl_datatype_t data_type) noexcept
{  // TODO
    return Z_OK;
}
}  // namespace ccl
}  // namespace zbccl