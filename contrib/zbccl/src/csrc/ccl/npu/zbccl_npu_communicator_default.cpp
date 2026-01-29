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
#include "zbccl_npu_communicator_default.h"
#include "zbccl_npu_op_allreduce.h"
#include "zbccl_npu_op_reducescatter.h"
#include "zbccl_npu_operators.h"
#include "dl_cann_api.h"
#include "acl/acl.h"

namespace zbccl {
namespace ccl {

using namespace underapi;

NpuCommunicatorDefault::NpuCommunicatorDefault(const CommGroupOptions &options, bool isWorldGroup,
                                               const CommunicatorPtr &worldGroup)
    : Communicator(options, isWorldGroup, worldGroup)
{}

ZResult NpuCommunicatorDefault::Initialize() noexcept
{
    /* get ffts address */
    uint32_t len = 0;
    auto result = DlCannApi::RtGetC2cCtrlAddr(&groupInfo_.fftsConfig, &len);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("get c2c ctrl addr failed, result: " << result);
        return Z_FFTS_INIT_FAILED;
    }

    /* copy group info to meta area of communicator from host to device */
    ZBCCL_ASSERT_RETURN(sizeof(CommGroupInfo) == groupInfo_.sizeForCommGroupInfo, Z_ERROR);
    result = DlCannApi::AclrtMemcpy(reinterpret_cast<void *>(groupInfo_.myMetaGva), sizeof(CommGroupInfo), &groupInfo_,
                                     sizeof(CommGroupInfo), ACL_MEMCPY_HOST_TO_DEVICE);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("get c2c ctrl addr failed, result: " << result);
        return Z_FFTS_INIT_FAILED;
    }

    return Z_OK;
}

void NpuCommunicatorDefault::UnInitialize() noexcept
{
    // TODO
}

void NpuCommunicatorDefault::ConstructCommGroupInfo(const CommGroupOptions &options) noexcept
{
    groupInfo_.groupSize = options.groupSize;
    groupInfo_.myGroupRank = options.myGroupRank;
    groupInfo_.myMetaGva = options.myMetaGva;
    groupInfo_.myParamDataGva = options.myParamDataGva;
    groupInfo_.myAddressExchangeGva = options.myAddressExchangeGva;
    groupInfo_.sizeForCommGroupInfo = options.sizeForCommGroupInfo;
    groupInfo_.sizeForParam = options.sizeForParam;
    groupInfo_.sizeForExchangeAddress = options.sizeForExchangeAddress;
    groupInfo_.fftsConfig = options.fftsConfig;
    groupInfo_.localDeviceMemSize = options.localDeviceMemSize;
}

int32_t NpuCommunicatorDefault::AllReduce(const void *send_buff, void *recv_buff, size_t count,
                                          zbccl_datatype_t data_type, zbccl_reduce_op_t op, aclrtStream stream) noexcept
{
    auto groupInfo = GetMetaInfo();
    zbccl::underapi::DlCannApi::AclrtMemset(reinterpret_cast<void *>(groupInfo.myAddressExchangeGva),
        groupInfo.sizeForExchangeAddress, 0, groupInfo.sizeForExchangeAddress);
    zbccl::underapi::DlCannApi::AclrtMemset(recv_buff, count, 0, count);
    auto ret = ZBCCLAllReduce(send_buff, recv_buff, count, data_type, stream, op, groupInfo);
    return ret;
}

int32_t NpuCommunicatorDefault::ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count,
                                              zbccl_datatype_t data_type, zbccl_reduce_op_t op,
                                              aclrtStream stream) noexcept
{
    auto groupInfo = GetMetaInfo();
    zbccl::underapi::DlCannApi::AclrtMemset(reinterpret_cast<void *>(groupInfo.myAddressExchangeGva),
        groupInfo.sizeForExchangeAddress, 0, groupInfo.sizeForExchangeAddress);
    zbccl::underapi::DlCannApi::AclrtMemset(recv_buff, recv_count, 0, recv_count);
    auto ret = ZBCCLReduceScatter(send_buff, recv_buff, recv_count, data_type, stream, op, groupInfo);
    return ret;
}

int32_t NpuCommunicatorDefault::AllGather(const void *send_buff, void *recv_buff, size_t send_count,
                                          zbccl_datatype_t data_type, aclrtStream stream) noexcept
{
    auto groupInfo = GetMetaInfo();
    zbccl::underapi::DlCannApi::AclrtMemset(reinterpret_cast<void *>(groupInfo.myAddressExchangeGva),
        groupInfo.sizeForExchangeAddress, 0, groupInfo.sizeForExchangeAddress);
    return ZBCCLOpAllGather(send_buff, recv_buff, send_count, data_type, stream, groupInfo);
}

int32_t NpuCommunicatorDefault::All2All(const void *sendBuff, void *recvBuff, uint64_t data_count,
                                        zbccl_datatype_t dataType, uint64_t stride_count, uint8_t repeat,
                                        aclrtStream stream) noexcept
{
    // TODO
    return Z_OK;
}

int32_t NpuCommunicatorDefault::DispatchNormalNotify(const zbccl_tensor_info_t *sendTokensPerExpert, int64_t sendCount,
                                                     int64_t topKNum, const zbccl_tensor_info_t *recvBuff,
                                                     int64_t *totalRecvTokens,
                                                     const zbccl_tensor_info_t *recvTokensPerExpert,
                                                     const zbccl_tensor_info_t *pushTargetOffset,
                                                     const zbccl_tensor_info_t *balanceMatrix,
                                                     int64_t flags) noexcept
{
    // TODO
    return Z_OK;
}

int32_t NpuCommunicatorDefault::DispatchNormalLayout(const zbccl_tensor_info_t *topkIndex, int64_t tokens,
                                                     int64_t expertNum, int64_t topkNum,
                                                     const zbccl_tensor_info_t *tokensPerRank,
                                                     const zbccl_tensor_info_t *tokensPerExpert,
                                                     const zbccl_tensor_info_t *isTokenInRank,
                                                     const zbccl_tensor_info_t *sendTokensIndex, aclrtStream stream,
                                                     int64_t flags) noexcept
{
    // return ZBCCL_OP_DispatchLayout(topkIndex, tokens, expertNum, topkNum, tokensPerRank, tokensPerExpert,
                                //    isTokenInRank, sendTokensIndex, stream, GetMetaInfo(), flags);
    return Z_OK;
}

int32_t NpuCommunicatorDefault::DispatchNormal(const zbccl_tensor_info_t *srcTokens,
                                               const zbccl_tensor_info_t *topkIndex,
                                               const zbccl_tensor_info_t *sendTokensIndex,
                                               const zbccl_tensor_info_t *pushTargetOffset, int64_t expertNum,
                                               zbccl_quant_mode_t quantMode, const zbccl_tensor_info_t *destTokens,
                                               const zbccl_tensor_info_t *destScale, zbccl_comm_t comm,
                                               aclrtStream stream, int64_t flags) noexcept
{
    // TODO
    return Z_OK;
}

int32_t NpuCommunicatorDefault::CombineNormal(const zbccl_tensor_info_t *srcTokens,
                                              const zbccl_tensor_info_t *srcTokensPerEp,
                                              const zbccl_tensor_info_t *topKWeight,
                                              const zbccl_tensor_info_t *topkIndex,
                                              const zbccl_tensor_info_t *sendTokensIndex,
                                              const zbccl_tensor_info_t *balanceMatrix, uint16_t expertNum,
                                              const zbccl_tensor_info_t *destTokens, zbccl_comm_t comm,
                                              aclrtStream stream, int64_t flags) noexcept
{
    // TODO
    return Z_OK;
}
}  // namespace ccl
}  // namespace zbccl