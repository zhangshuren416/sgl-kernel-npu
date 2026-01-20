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

ZBCCL_API int32_t zbccl_create(zbccl_ccl_options_t *options, zbccl_comm_t *comm)
{
    ZBCCL_VALIDATE_RETURN(options != nullptr, "Create zbccl communicator failed as options is null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "Create zbccl communicator failed as comm is null", Z_INVALID_PARAM);

    ZBCCL_LOG_INFO("options dump, " << (*options));

    auto &state = ZBCCLInitState::Instance();

    if (!state.Bootstrapped()) {
        ZBCCL_LOG_ERROR("Create zbccl communicator failed as not bootstrapped");
        return Z_NOT_BOOTSTRAPPED;
    } else if (options->isWorldGroup == 1 && state.WorldSize() != options->groupSize) {
        ZBCCL_LOG_ERROR("Create zbccl communicator failed as world size <"
                        << options->groupSize << "> is not equal to bootstrap's world size <" << state.WorldSize()
                        << ">");
        return Z_NOT_BOOTSTRAPPED;
    } else if (options->isWorldGroup == 0 && state.WorldSize() < options->groupSize) {
        ZBCCL_LOG_ERROR("Create zbccl communicator failed as world size <"
                        << options->groupSize << "> is bigger than bootstrap's world size <" << state.WorldSize()
                        << ">");
        return Z_NOT_BOOTSTRAPPED;
    }

    /* create one communicator */
    auto result = ZBCCLComm::Create(*options, comm, state.WorldSize(), state.WorldRankId(), state.DeviceId());
    if (result != Z_OK) {
        return result;
    }

    /* update init state */
    state.CommunicatorCreated();

    return Z_OK;
}

ZBCCL_API int32_t zbccl_destroy(zbccl_comm_t *comm, uint32_t flags)
{
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "Create zbccl communicator failed as comm is null", Z_INVALID_PARAM);

    /* destroy one */
    auto result = ZBCCLComm::Destroy(comm, flags);
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
    auto innerComm = reinterpret_cast<ZBCCLComm *>(comm);
    return innerComm->AllReduce(send_buff, recv_buff, count, data_type, op);
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
    auto innerComm = reinterpret_cast<ZBCCLComm *>(comm);
    return innerComm->ReduceScatter(send_buff, recv_buff, recv_count, data_type, op);
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
    auto innerComm = reinterpret_cast<ZBCCLComm *>(comm);
    return innerComm->AllGather(send_buff, recv_buff, send_count, data_type);
}

#ifdef __cplusplus
}
#endif
