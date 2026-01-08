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
#include "zbccl_operations.h"
#include "zbccl_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t zbccl_init() {
    // TODO
    return zbccl::ZResultErrorCode::Z_OK;
}

int32_t zbccl_all_reduce(const void *send_buff, void *recv_buff, size_t count, zbccl_datatype_t data_type,
                         zbccl_reduce_op_t op, zbccl_comm_t comm, aclrtStream stream) {
    // TODO
    return zbccl::ZResultErrorCode::Z_OK;
}

int32_t zbccl_reduce_scatter(const void *send_buff, void *recv_buff, size_t recv_count, zbccl_datatype_t data_type,
                             zbccl_reduce_op_t op, zbccl_comm_t comm, aclrtStream stream) {
    // TODO
    return zbccl::ZResultErrorCode::Z_OK;
}

int32_t zbccl_all_gather(const void *send_buff, void *recv_buff, size_t send_count, zbccl_datatype_t data_type,
                         zbccl_comm_t comm, aclrtStream stream) {
    // TODO
    return zbccl::ZResultErrorCode::Z_OK;
}

#ifdef __cplusplus
}
#endif
