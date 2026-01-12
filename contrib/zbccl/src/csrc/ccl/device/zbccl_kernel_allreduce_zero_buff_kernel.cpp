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
#include "zbccl_kernel_allreduce_zero_buff_kernel.h"

using namespace zbccl;
using namespace zbccl::allreduce;

extern "C" __global__ __aicore__ void ShmemZeroBuffAllReduce(
    GM_ADDR input, GM_ADDR output, GM_ADDR gva,
    uint64_t fftsAddr, uint32_t dataType, uint32_t totalLength,
    int teamId, uint32_t reduceOp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t magic = 1;
    uint32_t rank = shmem_team_my_pe(teamId);
    uint32_t rankSize = shmem_team_n_pes(teamId);
    ZCCLDataType zcclDataType = static_cast<ZCCLDataType>(dataType);
    switch (zcclDataType) {
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32: {
            ZeroBuffAllReduceKernel<int32_t> op;
            op.Init(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32: {
            ZeroBuffAllReduceKernel<float> op;
            op.Init(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_FP16: {
            ZeroBuffAllReduceKernel<float16_t> op;
            op.Init(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_BFP16: {
            ZeroBuffAllReduceKernel<bfloat16_t> op;
            op.Init(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        default:
            return;
    }
}