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
#ifndef ZBCCL_COMMUNICATOR_DEFAULT_H
#define ZBCCL_COMMUNICATOR_DEFAULT_H

#include "zbccl_communicator.h"

namespace zbccl {
namespace ccl {

class ZBCCLCommDefault : public ZBCCLComm
{
public:
    ZBCCLCommDefault(const ZBCommOptions &options, bool isWorldGroup, const ZBCCLCommPtr &worldGroup);
    ~ZBCCLCommDefault() override
    {
        UnInitialize();
    }

    ZResult Initialize() noexcept override;

    void UnInitialize() noexcept override;

    int32_t AllReduce(const void *send_buff, void *recv_buff, size_t count, zbccl_datatype_t data_type,
                      zbccl_reduce_op_t op) noexcept override;

    int32_t ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, zbccl_datatype_t data_type,
                          zbccl_reduce_op_t op) noexcept override;

    int32_t AllGather(const void *send_buff, void *recv_buff, size_t send_count,
                      zbccl_datatype_t data_type) noexcept override;

private:
    void *kernelMetaH2DArea_ = nullptr;       /* dram space for exchange the meta of zbccl operators */
    void *kernelMetaH2DAreaDevice_ = nullptr; /* the device ptr of kernelMetaH2DArea */
};
using ZBCCLCommDefaultPtr = ZRef<ZBCCLCommDefault>;
}  // namespace ccl
}  // namespace zbccl

#endif  // ZBCCL_COMMUNICATOR_DEFAULT_H
