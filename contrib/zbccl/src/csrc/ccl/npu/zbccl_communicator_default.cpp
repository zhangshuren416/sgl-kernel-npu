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
                                    zbccl_reduce_op_t op, aclrtStream stream) noexcept
{
    // TODO
    return Z_OK;
}

int32_t ZBCCLCommDefault::ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count,
                                        zbccl_datatype_t data_type, zbccl_reduce_op_t op, aclrtStream stream) noexcept
{
    // TODO
    return Z_OK;
}

int32_t ZBCCLCommDefault::AllGather(const void *send_buff, void *recv_buff, size_t send_count,
                                    zbccl_datatype_t data_type, aclrtStream stream) noexcept
{
    // TODO
    ZBCCL_LOG_INFO("inner allgather");
    return Z_OK;
}

int32_t ZBCCLCommDefault::All2All(const void *sendBuff, void *recvBuff, uint64_t data_count, zbccl_datatype_t dataType,
                                  uint64_t stride_count, uint8_t repeat, aclrtStream stream) noexcept
{
    // TODO
    return Z_OK;
}

int32_t ZBCCLCommDefault::DispatchNormalNotify(const zbccl_tensor_info_t *sendTokensPerExpert, int64_t sendCount,
                                               int64_t topKNum, const zbccl_tensor_info_t *recvBuff,
                                               int64_t *totalRecvTokens, const zbccl_tensor_info_t *recvTokensPerExpert,
                                               const zbccl_tensor_info_t *pushTargetOffset, int64_t flags) noexcept
{
    // TODO
    return Z_OK;
}

int32_t ZBCCLCommDefault::DispatchNormalLayout(const zbccl_tensor_info_t *topkIndex, int64_t tokens, int64_t expertNum,
                                               int64_t topkNum, const zbccl_tensor_info_t *tokensPerRank,
                                               const zbccl_tensor_info_t *tokensPerExpert,
                                               const zbccl_tensor_info_t *isTokenInRank,
                                               const zbccl_tensor_info_t *tokenIndex, zbccl_comm_t comm,
                                               aclrtStream stream, int64_t flags) noexcept
{
    // TODO
    return Z_OK;
}

int32_t ZBCCLCommDefault::DispatchNormal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *topkIndex,
                                         const zbccl_tensor_info_t *sendTokensIndex,
                                         const zbccl_tensor_info_t *pushTargetOffset, int64_t expertNum,
                                         zbccl_quant_mode_t quantMode, const zbccl_tensor_info_t *destTokens,
                                         const zbccl_tensor_info_t *destScale, zbccl_comm_t comm, aclrtStream stream,
                                         int64_t flags) noexcept
{
    // TODO
    return Z_OK;
}

int32_t ZBCCLCommDefault::CombineNormal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *srcTokensPerEp,
                                        const zbccl_tensor_info_t *topKWeight, const zbccl_tensor_info_t *topkIndex,
                                        const zbccl_tensor_info_t *sendTokensIndex, uint16_t expertNum,
                                        const zbccl_tensor_info_t *destTokens, zbccl_comm_t comm, aclrtStream stream,
                                        int64_t flags) noexcept
{
    // TODO
    return Z_OK;
}
}  // namespace ccl
}  // namespace zbccl