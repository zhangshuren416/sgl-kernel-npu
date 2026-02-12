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
#ifndef ZBCCL_COMMUNICATOR_DUMMY_H
#define ZBCCL_COMMUNICATOR_DUMMY_H

#include "zbccl_communicator.h"

namespace zbccl {
namespace ccl {

/**
 * This is for unit test, a dummy communicator doesn't depend on any hardware
 */
class CommunicatorDummy : public Communicator
{
public:
    CommunicatorDummy(const CommGroupOptions &options, bool isWorldGroup, const CommunicatorPtr &worldGroup)
        : Communicator(options, isWorldGroup, worldGroup)
    {}

    ~CommunicatorDummy() override = default;

    ZResult Initialize() noexcept override
    {
        return Z_OK;
    }

    void UnInitialize() noexcept override {}

    void ConstructCommGroupInfo(const CommGroupOptions &opt) noexcept override {}

    ZResult AssignGatherGroupId(AutoReleaseGroupId &id) noexcept override
    {
        return Z_OK;
    }

    int32_t AllReduce(const void *send_buff, void *recv_buff, void *buffer, size_t count, size_t buf_cnt, zbccl_datatype_t data_type,
                      zbccl_reduce_op_t op, aclrtStream stream) noexcept override
    {
        return Z_OK;
    }

    int32_t ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, zbccl_datatype_t data_type,
                          zbccl_reduce_op_t op, aclrtStream stream) noexcept override
    {
        return Z_OK;
    }

    int32_t AllGather(const void *send_buff, void *recv_buff, size_t send_count, zbccl_datatype_t data_type,
                      aclrtStream stream) noexcept override
    {
        return Z_OK;
    }

    int32_t All2All(const void *sendBuff, void *recvBuff, uint64_t data_count, zbccl_datatype_t dataType,
                    uint64_t stride_count, uint8_t repeat, aclrtStream stream) noexcept
    {
        return Z_OK;
    }

    int32_t DispatchNormalNotify(const zbccl_tensor_info_t *sendTokensPerExpert, int64_t sendCount, int64_t topKNum,
                                 const zbccl_tensor_info_t *recvBuff, const zbccl_tensor_info_t *totalRecvTokens,
                                 const zbccl_tensor_info_t *recvTokensPerExpert,
                                 const zbccl_tensor_info_t *pushTargetOffset, const zbccl_tensor_info_t *balanceMatrix,
                                 aclrtStream stream, int64_t flags) noexcept
    {
        return Z_OK;
    }

    int32_t DispatchNormalLayout(const zbccl_tensor_info_t *topkIndex, int64_t tokens, int64_t expertNum,
                                 int64_t topkNum, const zbccl_tensor_info_t *tokensPerRank,
                                 const zbccl_tensor_info_t *tokensPerExpert, const zbccl_tensor_info_t *isTokenInRank,
                                 const zbccl_tensor_info_t *sendTokensIndex, const zbccl_tensor_info_t *notifySendData,
                                 aclrtStream stream, int64_t flags) noexcept
    {
        return Z_OK;
    }

    int32_t DispatchNormal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *topkIndex,
                           const zbccl_tensor_info_t *sendTokensIndex, const zbccl_tensor_info_t *pushTargetOffset,
                           const zbccl_tensor_info_t *balanceMatrix, int64_t expertNum, zbccl_quant_mode_t quantMode,
                           const zbccl_tensor_info_t *destTokens, const zbccl_tensor_info_t *destScale,
                           aclrtStream stream, int64_t flags) noexcept
    {
        return Z_OK;
    }

    int32_t CombineNormal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *srcTokensPerEp,
                          const zbccl_tensor_info_t *topKWeight, const zbccl_tensor_info_t *topkIndex,
                          const zbccl_tensor_info_t *sendTokensIndex, const zbccl_tensor_info_t *balanceMatrix,
                          uint16_t expertNum, const zbccl_tensor_info_t *destTokens,
                          aclrtStream stream, int64_t flags) noexcept
    {
        return Z_OK;
    }
};
}  // namespace ccl
}  // namespace zbccl

#endif  // ZBCCL_COMMUNICATOR_DUMMY_H
