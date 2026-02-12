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
#include "zbccl_common_includes.h"
#include "zbccl_communicator.h"
#include "zbccl_init_state.h"

using namespace zbccl;
using namespace zbccl::ccl;

#ifdef __cplusplus
extern "C" {
#endif

ZBCCL_API int32_t zbccl_comm_create(zbccl_comm_options_t *options, zbccl_comm_t *comm)
{
    ZBCCL_VALIDATE_RETURN(options != nullptr, "Create communicator failed as options is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "Create communicator failed as comm is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(options->name != nullptr, "Create communicator failed as name is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(strlen(options->name) != 0, "Create communicator failed as name is empty", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(
        strlen(options->name) < ZBCCL_COMM_NAME_MAX,
        "Create communicator failed as name is too long, which should be less than " << ZBCCL_COMM_NAME_MAX,
        Z_INVALID_PARAM);

    ZBCCL_LOG_INFO("options dump, " << (*options));

    auto &state = ZBCCLInitState::Instance();

    if (!state.Bootstrapped()) {
        ZBCCL_LOG_ERROR("Create communicator failed as not bootstrapped");
        return Z_NOT_BOOTSTRAPPED;
    } else if (options->isWorldGroup == 1 && state.ext_.worldSize != options->groupSize) {
        ZBCCL_LOG_ERROR("Create communicator failed as world size "
                        << options->groupSize << " is not equal to bootstrap's world size " << state.ext_.worldSize);
        return Z_NOT_BOOTSTRAPPED;
    } else if (options->isWorldGroup == 0 && state.ext_.worldSize < options->groupSize) {
        ZBCCL_LOG_ERROR("Create communicator failed as world size "
                        << options->groupSize << " is bigger than bootstrap's world size " << state.ext_.worldSize);
        return Z_NOT_BOOTSTRAPPED;
    }

    /* create one comm */
    auto result = Communicator::Create(*options, comm, state.ext_);
    if (result != Z_OK) {
        return result;
    }

    /* update init state */
    state.CommunicatorCreated(1);

    return Z_OK;
}

ZBCCL_API int32_t zbccl_comm_get_property(zbccl_comm_t comm, zbccl_comm_property_t *property)
{
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "Get property failed as comm is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(property != nullptr, "Get property failed as property is null", Z_INVALID_PARAM);

    return Communicator::GetCommProperty(comm, property);
}

ZBCCL_API uintptr_t zbccl_comm_get_global()
{
    uintptr_t comm;
    auto result = Communicator::GetGlobalComm(comm);
    if (result != Z_OK) {
        return reinterpret_cast<uintptr_t>(nullptr);
    }
    return comm;
}

ZBCCL_API zbccl_comm_t zbccl_comm_get_by_name(const char *name)
{
    ZBCCL_VALIDATE_RETURN(name != nullptr, "Get communicator failed as name is null", nullptr);

    zbccl_comm_t comm = nullptr;
    auto result = Communicator::Lookup(std::string(name), &comm);
    if (result != Z_OK) {
        return nullptr;
    }

    return comm;
}

ZBCCL_API int32_t zbccl_comm_destroy(zbccl_comm_t comm, uint32_t flags)
{
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "Destroy communicator failed as comm is null", Z_INVALID_PARAM);

    /* destroy one */
    auto result = Communicator::Destroy(comm, flags);
    if (result != Z_OK) {
        return result;
    }

    /* update init state */
    ZBCCLInitState::Instance().CommunicatorDestroy(1);

    return Z_OK;
}

ZBCCL_API void zbccl_comm_destroy_all(uint32_t flags)
{
    (void)flags;
    ZBCCL_LOG_WARN("destroy all comm group");
    Communicator::DestroyAll();
}

ZBCCL_API int32_t zbccl_all_reduce(const void *send_buff, void *recv_buff, void *buffer, size_t count, size_t buf_cnt, zbccl_datatype_t data_type,
                                   zbccl_reduce_op_t op, zbccl_comm_t comm, aclrtStream stream)
{
    if (send_buff == nullptr) {
        return Z_OK;
    }
    ZBCCL_VALIDATE_RETURN(recv_buff != nullptr, "AllReduce failed as recv_buff is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(count > 0, "AllReduce failed as count " << count << " is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(data_type >= 0 && data_type < ZBCCL_DATA_TYPE_BUTT,
                          "AllReduce failed as data_type " << data_type << " is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(op >= 0 && op < ZBCCL_REDUCE_BUTT, "AllReduce failed as op " << op << " is invalid",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "AllReduce failed as comm is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(buffer != nullptr, "Allreduce tmp buffer is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(buf_cnt > 0, "AllReduce buf cnt is invalid", Z_INVALID_PARAM);

    /* covert inner object ptr and execute op */
    auto innerComm = reinterpret_cast<Communicator *>(comm);
    return innerComm->AllReduce(send_buff, recv_buff, buffer, count, buf_cnt, data_type, op, stream);
}

ZBCCL_API int32_t zbccl_reduce_scatter(const void *send_buff, void *recv_buff, size_t recv_count,
                                       zbccl_datatype_t data_type, zbccl_reduce_op_t op, zbccl_comm_t comm,
                                       aclrtStream stream)
{
    if (send_buff == nullptr) {
        return Z_OK;
    }
    ZBCCL_VALIDATE_RETURN(recv_buff != nullptr, "ReduceScatter failed as recv_buff is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(recv_count > 0, "ReduceScatter failed as recv_count " << recv_count << " is invalid",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(data_type >= 0 && data_type < ZBCCL_DATA_TYPE_BUTT,
                          "ReduceScatter failed as data_type " << data_type << " is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(op >= 0 && op < ZBCCL_REDUCE_BUTT, "ReduceScatter failed as op " << op << " is invalid",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "ReduceScatter failed as comm is null", Z_INVALID_PARAM);

    /* covert inner object ptr and execute op */
    auto innerComm = reinterpret_cast<Communicator *>(comm);
    return innerComm->ReduceScatter(send_buff, recv_buff, recv_count, data_type, op, stream);
}

ZBCCL_API int32_t zbccl_all_gather(const void *send_buff, void *recv_buff, size_t send_count,
                                   zbccl_datatype_t data_type, zbccl_comm_t comm, aclrtStream stream)
{
    if (send_buff == nullptr) {
        return Z_OK;
    }
    ZBCCL_VALIDATE_RETURN(recv_buff != nullptr, "AllGather failed as recv_buff is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(send_count > 0, "AllGather failed as send_count " << send_count << " is invalid",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(data_type >= 0 && data_type < ZBCCL_DATA_TYPE_BUTT,
                          "AllGather failed as data_type " << data_type << " is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "AllGather failed as comm is null", Z_INVALID_PARAM);

    /* covert inner object ptr and execute op */
    auto innerComm = reinterpret_cast<Communicator *>(comm);
    return innerComm->AllGather(send_buff, recv_buff, send_count, data_type, stream);
}

ZBCCL_API int32_t zbccl_all_to_all(const void *sendBuff, void *recvBuff, uint64_t data_count, zbccl_datatype_t dataType,
                                   uint64_t stride_count, uint8_t repeat, zbccl_comm_t comm, aclrtStream stream)
{
    return Z_OK;
}

ZBCCL_API int32_t zbccl_dispatch_normal_notify(const zbccl_tensor_info_t *sendTokensPerExpert, int64_t sendCount,
                                               int64_t topKNum, const zbccl_tensor_info_t *recvBuff,
                                               const zbccl_tensor_info_t *totalRecvTokens,
                                               const zbccl_tensor_info_t *recvTokensPerExpert,
                                               const zbccl_tensor_info_t *pushTargetOffset,
                                               const zbccl_tensor_info_t *balanceMatrix, zbccl_comm_t comm,
                                               aclrtStream stream, int64_t flags)
{
    ZBCCL_VALIDATE_RETURN(sendTokensPerExpert != nullptr, "NotifyDispatch failed as sendTokensPerExpert is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(sendCount > 0, "NotifyDispatch failed as sendCount " << sendCount << " is invalid",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(topKNum > 0, "NotifyDispatch failed as topKNum " << topKNum << " is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(recvBuff != nullptr, "NotifyDispatch failed as recvBuff is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(totalRecvTokens != nullptr, "NotifyDispatch failed as totalRecvTokens is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(recvTokensPerExpert != nullptr, "NotifyDispatch failed as recvTokensPerExpert is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(pushTargetOffset != nullptr, "NotifyDispatch failed as pushTargetOffset is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(balanceMatrix != nullptr, "NotifyDispatch failed as balanceMatrix is null", Z_INVALID_PARAM);

    /* covert inner object ptr and execute op */
    auto innerComm = reinterpret_cast<Communicator *>(comm);
    return innerComm->DispatchNormalNotify(sendTokensPerExpert, sendCount, topKNum, recvBuff, totalRecvTokens, recvTokensPerExpert,
                                           pushTargetOffset, balanceMatrix, stream, flags);
}

ZBCCL_API int32_t zbccl_dispatch_normal_layout(const zbccl_tensor_info_t *topkIndex, int64_t tokens, int64_t expertNum,
                                               int64_t topkNum, const zbccl_tensor_info_t *tokensPerRank,
                                               const zbccl_tensor_info_t *tokensPerExpert,
                                               const zbccl_tensor_info_t *isTokenInRank,
                                               const zbccl_tensor_info_t *sendTokensIndex,
                                               const zbccl_tensor_info_t *notifySendData, zbccl_comm_t comm,
                                               aclrtStream stream, int64_t flags)
{
    ZBCCL_VALIDATE_RETURN(topkIndex != nullptr, "DispatchLayout failed as topkIndex is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(tokensPerRank != nullptr, "DispatchLayout failed as tokensPerRank is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(tokensPerExpert != nullptr, "DispatchLayout failed as tokensPerExpert is null",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(isTokenInRank != nullptr, "DispatchLayout failed as isTokenInRank is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(sendTokensIndex != nullptr, "DispatchLayout failed as sendTokensIndex is null",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(notifySendData != nullptr, "DispatchLayout failed as notifySendData is null",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(tokens >= 0, "DispatchLayout failed as tokens " << tokens << " is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(expertNum > 0, "DispatchLayout failed as expertNum " << expertNum << " is invalid",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(topkNum > 0, "DispatchLayout failed as topkNum " << topkNum << " is invalid",
                          Z_INVALID_PARAM);

    /* covert inner object ptr and execute op */
    auto innerComm = reinterpret_cast<Communicator *>(comm);
    return innerComm->DispatchNormalLayout(topkIndex, tokens, expertNum, topkNum, tokensPerRank, tokensPerExpert,
                                           isTokenInRank, sendTokensIndex, notifySendData, stream, flags);
}

ZBCCL_API int32_t zbccl_dispatch_normal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *topkIndex,
                                        const zbccl_tensor_info_t *sendTokensIndex,
                                        const zbccl_tensor_info_t *pushTargetOffset,
                                        const zbccl_tensor_info_t *balanceMatrix, int64_t expertNum,
                                        zbccl_quant_mode_t quantMode, const zbccl_tensor_info_t *destTokens,
                                        const zbccl_tensor_info_t *destScale, zbccl_comm_t comm, aclrtStream stream,
                                        int64_t flags)
{
    ZBCCL_VALIDATE_RETURN(srcTokens != nullptr, "DispatchNormal failed as srcTokens is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(topkIndex != nullptr, "DispatchNormal failed as topkIndex is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(sendTokensIndex != nullptr, "DispatchNormal failed as sendTokensIndex is null",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(pushTargetOffset != nullptr, "DispatchNormal failed as pushTargetOffset is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(balanceMatrix != nullptr, "DispatchNormal failed as balanceMatrix is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(sendTokensIndex != nullptr, "DispatchNormal failed as sendTokensIndex is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(destTokens != nullptr, "DispatchNormal failed as destTokens is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(destScale != nullptr, "DispatchNormal failed as destScale is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(expertNum > 0, "DispatchNormal failed as expertNum " << expertNum << " is invalid",
                          Z_INVALID_PARAM);

    /* covert inner object ptr and execute op */
    auto innerComm = reinterpret_cast<Communicator *>(comm);
    return innerComm->DispatchNormal(srcTokens, topkIndex, sendTokensIndex, pushTargetOffset, balanceMatrix, expertNum,
                                    quantMode, destTokens, destScale, stream, flags);
}

ZBCCL_API int32_t zbccl_combine_normal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *srcTokensPerEp,
                                       const zbccl_tensor_info_t *topKWeight, const zbccl_tensor_info_t *topkIndex,
                                       const zbccl_tensor_info_t *sendTokensIndex,
                                       const zbccl_tensor_info_t *balanceMatrix, uint16_t expertNum,
                                       const zbccl_tensor_info_t *destTokens, zbccl_comm_t comm, aclrtStream stream,
                                       int64_t flags)
{
    ZBCCL_VALIDATE_RETURN(srcTokens != nullptr, "CombineNormal failed as srcTokens is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(srcTokensPerEp != nullptr, "CombineNormal failed as srcTokensPerEp is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(topKWeight != nullptr, "CombineNormal failed as topKWeight is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(topkIndex != nullptr, "CombineNormal failed as topkIndex is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(sendTokensIndex != nullptr, "CombineNormal failed as sendTokensIndex is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(balanceMatrix != nullptr, "CombineNormal failed as balanceMatrix is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(destTokens != nullptr, "CombineNormal failed as destTokens is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(expertNum > 0, "CombineNormal failed as expertNum " << expertNum << " is invalid",
                          Z_INVALID_PARAM);

    /* covert inner object ptr and execute op */
    auto innerComm = reinterpret_cast<Communicator *>(comm);
    return innerComm->CombineNormal(srcTokens, srcTokensPerEp, topKWeight, topkIndex, sendTokensIndex, balanceMatrix, expertNum,
                                    destTokens, stream, flags);
}

#ifdef __cplusplus
}
#endif
