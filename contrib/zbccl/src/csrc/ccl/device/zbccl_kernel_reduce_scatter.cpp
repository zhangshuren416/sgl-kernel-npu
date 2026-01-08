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

#include "zbccl_kernel_reduce_scatter.h"
using namespace zbccl;
using namespace zbccl::reducescatter;

extern "C" __global__ __aicore__ void ShmemReduceScatter(GM_ADDR input, GM_ADDR output, GM_ADDR gva,
                                                         uint64_t fftsAddr, uint32_t dataType, uint32_t totalLength,
                                                         int teamId, uint32_t reduceOp, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto tileData = reinterpret_cast<__gm__ ReduceScatterTilingData *>(tilingGM);
    uint32_t magic = 1;
    const int64_t aivNum = AscendC::GetBlockNum();
    uint32_t rank = shmem_team_my_pe(teamId);
    uint32_t rankSize = shmem_team_n_pes(teamId);
    bool lastLoop = true;
    ZCCLDataType zcclDataType = static_cast<ZCCLDataType>(dataType);
    switch (zcclDataType) {
        case ZCCLDataType::ZCCL_DATA_TYPE_INT8: {
            ReduceScatterKernel<int8_t> op;
            if (tileData->smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, lastLoop, magic, fftsAddr, reduceOp, tileData);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp, tileData);
            }
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_INT16: {
            ReduceScatterKernel<int16_t> op;
            if (tileData->smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, lastLoop, magic, fftsAddr, reduceOp, tileData);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp, tileData);
            }
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32: {
            ReduceScatterKernel<int32_t> op;
            if (tileData->smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, lastLoop, magic, fftsAddr, reduceOp, tileData);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp, tileData);
            }
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_INT64: {
            ReduceScatterKernel<int64_t> op;
            if (tileData->smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, lastLoop, magic, fftsAddr, reduceOp, tileData);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp, tileData);
            }
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32: {
            ReduceScatterKernel<float> op;
            if (tileData->smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, lastLoop, magic, fftsAddr, reduceOp, tileData);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp, tileData);
            }
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_FP16: {
            ReduceScatterKernel<float16_t> op;
            if (tileData->smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, lastLoop, magic, fftsAddr, reduceOp, tileData);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp, tileData);
            }
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_BFP16: {
            ReduceScatterKernel<bfloat16_t> op;
            if (tileData->smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, lastLoop, magic, fftsAddr, reduceOp, tileData);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp, tileData);
            }
        }
        default:
            return;
    }
}