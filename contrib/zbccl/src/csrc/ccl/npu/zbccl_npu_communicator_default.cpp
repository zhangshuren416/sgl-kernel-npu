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
    std::lock_guard<std::mutex> guard(mutex_);
    if (initialized_) {
        return Z_OK;
    }

    /* get ffts address */
    uint32_t len = 0;
    auto result = DlCannApi::RtGetC2cCtrlAddr(&groupInfo_.fftsConfig, &len);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("get c2c ctrl addr failed, result: " << result);
        return Z_FFTS_INIT_FAILED;
    }

    initialized_ = true;

    return Z_OK;
}

void NpuCommunicatorDefault::UnInitialize() noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_) {
        return;
    }

    initialized_ = false;
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
    groupInfo_.profilingGva = options.profilingGva;
    groupInfo_.sizeOfProfiling = options.sizeForProfiling;

    (void)DlCannApi::AclrtMemset(reinterpret_cast<void *>(groupInfo_.myMetaGva + offsetof(CommGroupInfo, vecCounter)),
                                 ZBCCL_SCALAR_CACHELINE_SIZE, 0, ZBCCL_SCALAR_CACHELINE_SIZE);
    (void)DlCannApi::AclrtMemset(reinterpret_cast<void *>(groupInfo_.myMetaGva + offsetof(CommGroupInfo, vecBarrier)),
                                 ZBCCL_SCALAR_CACHELINE_SIZE, 0, ZBCCL_SCALAR_CACHELINE_SIZE);
    (void)DlCannApi::AclrtMemset(reinterpret_cast<void *>(groupInfo_.myMetaGva + offsetof(CommGroupInfo, coreCounter)),
                                 ZBCCL_CORE_BARRIER_SIZE, 0, ZBCCL_CORE_BARRIER_SIZE);
    (void)DlCannApi::AclrtMemset(reinterpret_cast<void *>(groupInfo_.myMetaGva + offsetof(CommGroupInfo, coreBarrier)),
                                 ZBCCL_CORE_BARRIER_SIZE, 0, ZBCCL_CORE_BARRIER_SIZE);
    (void)DlCannApi::AclrtMemset(reinterpret_cast<void *>(groupInfo_.profilingGva),
                                 ZBCCL_CYCLE_PROFILING_SIZE, 0, ZBCCL_CYCLE_PROFILING_SIZE);
}

ZResult NpuCommunicatorDefault::AssignGatherGroupId(AutoReleaseGroupId &id) noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_) {
        ZBCCL_LOG_ERROR("Assign group id failed, as communicator is not initialized");
        return Z_NOT_INITIALIZED;
    }

    uniqueGroupId_.MoveIdAndGatheredInfo(id);

    /* assign peer info from group id class */
    auto &gatheredGroupInfo = uniqueGroupId_.GatheredGroupInfo();
    ZBCCL_ASSERT_RETURN(gatheredGroupInfo.size() == groupInfo_.groupSize, Z_ERROR);
    ZBCCL_ASSERT_RETURN(gatheredGroupInfo.size() <= ZBCCL_MAX_RANKS, Z_ERROR);

    for (uint64_t i = 0; i < gatheredGroupInfo.size(); ++i) {
        groupInfo_.peerGroupRank2WorldRank[i] = gatheredGroupInfo[i].myWorldRankId;
    }

    /* copy group info to meta area of communicator from host to device */
    ZBCCL_ASSERT_RETURN(sizeof(CommGroupInfo) == groupInfo_.sizeForCommGroupInfo, Z_ERROR);
    auto result = DlCannApi::AclrtMemcpy(reinterpret_cast<void *>(groupInfo_.myMetaGva), sizeof(CommGroupInfo), &groupInfo_,
                                    sizeof(CommGroupInfo), ACL_MEMCPY_HOST_TO_DEVICE);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("CommGroupInfo h2d copy failed, result: " << result);
        return Z_COMM_GROUP_H2D_FAILED;
    }

    ZBCCL_LOG_DEBUG("Dump groupId_ " << uniqueGroupId_ << ", groupInfo_: " << groupInfo_);
    return Z_OK;
}

int32_t NpuCommunicatorDefault::AllReduce(const void *send_buff, void *recv_buff, void *buffer, size_t count, size_t buf_cnt,
                                          zbccl_datatype_t data_type, zbccl_reduce_op_t op, aclrtStream stream) noexcept
{
    return ZBCCLOpAllReduce(send_buff, recv_buff, buffer, count, buf_cnt, data_type, stream, op, GetMetaInfo());
}

int32_t NpuCommunicatorDefault::ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count,
                                              zbccl_datatype_t data_type, zbccl_reduce_op_t op,
                                              aclrtStream stream) noexcept
{
    return ZBCCLOpReduceScatter(send_buff, recv_buff, recv_count, data_type, stream, op, GetMetaInfo());
}

int32_t NpuCommunicatorDefault::AllGather(const void *send_buff, void *recv_buff, size_t send_count,
                                          zbccl_datatype_t data_type, aclrtStream stream) noexcept
{
    return ZBCCLOpAllGather(send_buff, recv_buff, send_count, data_type, stream, GetMetaInfo());
}

int32_t NpuCommunicatorDefault::All2All(const void *sendBuff, void *recvBuff, uint64_t data_count,
                                        zbccl_datatype_t dataType, uint64_t stride_count, uint8_t repeat,
                                        aclrtStream stream) noexcept
{
    // TODO
    return Z_OK;
}

int32_t NpuCommunicatorDefault::DispatchNormalNotify(
    const zbccl_tensor_info_t *sendTokensPerExpert, int64_t sendCount, int64_t topKNum,
    const zbccl_tensor_info_t *recvBuff, const zbccl_tensor_info_t *totalRecvTokens,
    const zbccl_tensor_info_t *recvTokensPerExpert, const zbccl_tensor_info_t *pushTargetOffset,
    const zbccl_tensor_info_t *balanceMatrix, aclrtStream stream, int64_t flags) noexcept
{
    // get from env for balanceMatrix
    float factorHigh = Func::GetEnv("DEEPEP_BALANCE_FACTOR_HIGH", 1.2);
    float factorLow = Func::GetEnv("DEEPEP_BALANCE_FACTOR_LOW", 1.0);
    ZBCCL_CHECK_S(factorHigh > 1.1, "balance factor high need be large than 1.1");
    ZBCCL_CHECK_S(factorLow > 0.9, "balance factor low need be large than 0.9");
    return ZBCCLOpNotifyDispatch(sendTokensPerExpert, sendCount, topKNum, recvBuff, totalRecvTokens, recvTokensPerExpert,
                                 pushTargetOffset, balanceMatrix, factorHigh, factorLow, stream, GetMetaInfo(), flags);
}

int32_t NpuCommunicatorDefault::DispatchNormalLayout(
    const zbccl_tensor_info_t *topkIndex, int64_t tokens, int64_t expertNum, int64_t topkNum,
    const zbccl_tensor_info_t *tokensPerRank, const zbccl_tensor_info_t *tokensPerExpert,
    const zbccl_tensor_info_t *isTokenInRank, const zbccl_tensor_info_t *sendTokensIndex,
    const zbccl_tensor_info_t *notifySendData, aclrtStream stream, int64_t flags) noexcept
{
    return ZBCCLOpDispatchLayout(topkIndex, tokens, expertNum, topkNum, tokensPerRank, tokensPerExpert, isTokenInRank,
                                 sendTokensIndex, notifySendData, stream, GetMetaInfo(), flags);
}

int32_t NpuCommunicatorDefault::DispatchNormal(
    const zbccl_tensor_info_t *srcTokens,const zbccl_tensor_info_t *topkIndex,
    const zbccl_tensor_info_t *sendTokensIndex, const zbccl_tensor_info_t *pushTargetOffset,
    const zbccl_tensor_info_t *balanceMatrix, int64_t expertNum, zbccl_quant_mode_t quantMode,
    const zbccl_tensor_info_t *destTokens, const zbccl_tensor_info_t *destScale,
    aclrtStream stream, int64_t flags) noexcept
{
    // get env for balance dispatch
    bool enableBalance = Func::GetEnv("DEEPEP_ENABLE_REBALANCE", 0) > 0;
    return ZBCCLOpDispatchNormal(srcTokens, topkIndex, sendTokensIndex, pushTargetOffset, balanceMatrix, expertNum,
                                 quantMode, destTokens, destScale, enableBalance, stream, GetMetaInfo(), flags);
}

int32_t NpuCommunicatorDefault::CombineNormal(
    const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *srcTokensPerEp,
    const zbccl_tensor_info_t *topKWeight, const zbccl_tensor_info_t *topkIndex,
    const zbccl_tensor_info_t *sendTokensIndex, const zbccl_tensor_info_t *balanceMatrix, uint16_t expertNum,
    const zbccl_tensor_info_t *destTokens, aclrtStream stream, int64_t flags) noexcept
{
    // get env for balance dispatch
    bool enableBalance = Func::GetEnv("DEEPEP_ENABLE_REBALANCE", 0) > 0;
    return ZBCCLOpCombineNormal(srcTokens, srcTokensPerEp, topKWeight, topkIndex, sendTokensIndex, balanceMatrix,
                                expertNum, destTokens, enableBalance, stream, GetMetaInfo(), flags);
}
}  // namespace ccl
}  // namespace zbccl