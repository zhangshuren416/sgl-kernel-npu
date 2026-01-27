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
    state.CommunicatorCreated();

    return Z_OK;
}

ZBCCL_API int32_t zbccl_comm_get_property(zbccl_comm_t comm, zbccl_comm_property_t *property)
{
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "Get property as comm is null", Z_INVALID_PARAM);

    // TODO
    return Z_OK;
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
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "Create communicator failed as comm is null", Z_INVALID_PARAM);

    /* destroy one */
    auto result = Communicator::Destroy(comm, flags);
    if (result != Z_OK) {
        return result;
    }

    /* update init state */
    ZBCCLInitState::Instance().CommunicatorDestroy();

    return Z_OK;
}

ZBCCL_API int32_t zbccl_all_reduce(const void *send_buff, void *recv_buff, size_t count, zbccl_datatype_t data_type,
                                   zbccl_reduce_op_t op, zbccl_comm_t comm, aclrtStream stream)
{
    ZBCCL_VALIDATE_RETURN(send_buff != nullptr, "AllReduce failed, send_buff is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(recv_buff != nullptr, "AllReduce failed, recv_buff is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(count > 0, "AllReduce failed, count " << count << " is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(data_type >= 0 && data_type < ZBCCL_DATA_TYPE_BUTT,
                          "AllReduce failed, data_type " << data_type << " is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(op >= 0 && op < ZBCCL_REDUCE_BUTT, "AllReduce failed, op " << op << " is invalid",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "AllReduce failed, comm is null", Z_INVALID_PARAM);

    /* covert inner object ptr and execute op */
    auto innerComm = reinterpret_cast<Communicator *>(comm);
    return innerComm->AllReduce(send_buff, recv_buff, count, data_type, op, stream);
}

ZBCCL_API int32_t zbccl_reduce_scatter(const void *send_buff, void *recv_buff, size_t recv_count,
                                       zbccl_datatype_t data_type, zbccl_reduce_op_t op, zbccl_comm_t comm,
                                       aclrtStream stream)
{
    ZBCCL_VALIDATE_RETURN(send_buff != nullptr, "ReduceScatter failed, send_buff is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(recv_buff != nullptr, "ReduceScatter failed, recv_buff is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(recv_count > 0, "ReduceScatter failed, recv_count " << recv_count << " is invalid",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(data_type >= 0 && data_type < ZBCCL_DATA_TYPE_BUTT,
                          "ReduceScatter failed, data_type " << data_type << " is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(op >= 0 && op < ZBCCL_REDUCE_BUTT, "ReduceScatter failed, op " << op << " is invalid",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "ReduceScatter failed, comm is null", Z_INVALID_PARAM);

    /* covert inner object ptr and execute op */
    auto innerComm = reinterpret_cast<Communicator *>(comm);
    return innerComm->ReduceScatter(send_buff, recv_buff, recv_count, data_type, op, stream);
}

ZBCCL_API int32_t zbccl_all_gather(const void *send_buff, void *recv_buff, size_t send_count,
                                   zbccl_datatype_t data_type, zbccl_comm_t comm, aclrtStream stream)
{
    ZBCCL_VALIDATE_RETURN(send_buff != nullptr, "AllGather failed, send_buff is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(recv_buff != nullptr, "AllGather failed, recv_buff is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(send_count > 0, "AllGather failed, send_count " << send_count << " is invalid",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(data_type >= 0 && data_type < ZBCCL_DATA_TYPE_BUTT,
                          "AllGather failed, data_type " << data_type << " is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "AllGather failed, comm is null", Z_INVALID_PARAM);

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
                                               int64_t *totalRecvTokens, const zbccl_tensor_info_t *recvTokensPerExpert,
                                               const zbccl_tensor_info_t *pushTargetOffset, zbccl_comm_t comm,
                                               aclrtStream stream, int64_t flags)
{
    return Z_OK;
}

ZBCCL_API int32_t zbccl_dispatch_normal_layout(const zbccl_tensor_info_t *topkIndex, int64_t tokens, int64_t expertNum,
                                               int64_t topkNum, int64_t rankNum,
                                               const zbccl_tensor_info_t *tokensPerRank,
                                               const zbccl_tensor_info_t *tokensPerExpert,
                                               const zbccl_tensor_info_t *isTokenInRank,
                                               const zbccl_tensor_info_t *sendTokensIndex, aclrtStream stream,
                                               int64_t flags)
{
    return Z_OK;
}

ZBCCL_API int32_t zbccl_dispatch_normal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *topkIndex,
                                        const zbccl_tensor_info_t *sendTokensIndex,
                                        const zbccl_tensor_info_t *pushTargetOffset, int64_t expertNum,
                                        zbccl_quant_mode_t quantMode, const zbccl_tensor_info_t *destTokens,
                                        const zbccl_tensor_info_t *destScale, zbccl_comm_t comm, aclrtStream stream,
                                        int64_t flags)
{
    return Z_OK;
}

ZBCCL_API int32_t zbccl_combine_normal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *srcTokensPerEp,
                                       const zbccl_tensor_info_t *topKWeight, const zbccl_tensor_info_t *topkIndex,
                                       const zbccl_tensor_info_t *sendTokensIndex, uint16_t expertNum,
                                       const zbccl_tensor_info_t *destTokens, zbccl_comm_t comm, aclrtStream stream,
                                       int64_t flags)
{
    return Z_OK;
}

#ifdef __cplusplus
}
#endif
