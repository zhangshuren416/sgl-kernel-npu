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
#ifndef ZBCCL_OP_ALLREDUCE_H
#define ZBCCL_OP_ALLREDUCE_H

#include "zbccl_def.h"
#include "zbccl_communicator.h"

int32_t ZBCCLAllReduce(const void *inp, void *out, size_t numel, zbccl_datatype_t dataType,
                       aclrtStream stream, zbccl_reduce_op_t reduceOp, const CommGroupInfo &groupInfo);

#endif  // ZBCCL_OP_ALLREDUCE_H