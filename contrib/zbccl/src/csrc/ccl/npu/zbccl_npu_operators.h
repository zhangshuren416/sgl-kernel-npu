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
#ifndef ZBCCL_NPU_OPERATORS_H
#define ZBCCL_NPU_OPERATORS_H

#include "zbccl.h"
#include "zbccl_communicator.h"


int32_t ZBCCLOpAllGather(const void *sendBuff, void *recvBuff, size_t sendCount, zbccl_datatype_t dataType,
                         aclrtStream stream, const CommGroupInfo &groupInfo);

int32_t ZBCCLOpReduceScatter(const void *inp, void *out, size_t recvNumel, zbccl_datatype_t dataType,
                             aclrtStream stream, zbccl_reduce_op_t reduceOp, const CommGroupInfo &groupInfo);

int32_t ZBCCL_OP_DispatchLayout(const zbccl_tensor_info_t *topkIndex, int64_t tokens, int64_t expertNum,
                                int64_t topkNum, const zbccl_tensor_info_t *tokensPerRank,
                                const zbccl_tensor_info_t *tokensPerExpert,
                                const zbccl_tensor_info_t *isTokenInRank,
                                const zbccl_tensor_info_t *sendTokensIndex, aclrtStream stream,
                                const CommGroupInfo &groupInfo, int64_t flags);

#endif  // ZBCCL_NPU_OPERATORS_H
