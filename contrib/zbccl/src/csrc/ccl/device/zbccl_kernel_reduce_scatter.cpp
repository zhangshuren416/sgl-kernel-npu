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

__aicore__ inline size_t GetSizeFromTypeEnum(ZCCLDataType dtype)
{
    switch (dtype) {
        case ZCCLDataType::ZCCL_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_UINT8:
            return sizeof(uint8_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_UINT16:
            return sizeof(uint16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_UINT32:
            return sizeof(uint32_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_UINT64:
            return sizeof(uint64_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32:
            return sizeof(float);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP64:
            return sizeof(double);
        case ZCCLDataType::ZCCL_DATA_TYPE_BFP16:
            return sizeof(int16_t);
        default:
            return 0;
    }
}

extern "C" __global__ __aicore__ void ShmemReduceScatter(GM_ADDR input, GM_ADDR output, GM_ADDR gva,
                                                         uint64_t fftsAddr, uint32_t dataType, uint32_t totalLength,
                                                         int teamId, uint32_t reduceOp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t magic = 1;
    const int64_t aivNum = AscendC::GetBlockNum();
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t rank = shmem_team_my_pe(teamId);
    uint32_t rankSize = shmem_team_n_pes(teamId);
    ZCCLDataType zcclDataType = static_cast<ZCCLDataType>(dataType);
    size_t typeSize = GetSizeFromTypeEnum(zcclDataType);
    bool smallFlag = (totalLength >= BIG_DATA_SIZE / typeSize) ? false : true;
    switch (zcclDataType) {
        case ZCCLDataType::ZCCL_DATA_TYPE_INT8: {
            ReduceScatterKernel<int8_t> op;
            if (smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, totalLength, magic, fftsAddr, smallFlag, reduceOp);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            }
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_INT16: {
            ReduceScatterKernel<int16_t> op;
            if (smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, totalLength, magic, fftsAddr, smallFlag, reduceOp);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            }
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32: {
            ReduceScatterKernel<int32_t> op;
            if (smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, totalLength, magic, fftsAddr, smallFlag, reduceOp);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            }
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_INT64: {
            ReduceScatterKernel<int64_t> op;
            if (smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, totalLength, magic, fftsAddr, smallFlag, reduceOp);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            }
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32: {
            ReduceScatterKernel<float> op;
            if (smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, totalLength, magic, fftsAddr, smallFlag, reduceOp);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            }
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_FP16: {
            ReduceScatterKernel<float16_t> op;
            if (smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, totalLength, magic, fftsAddr, smallFlag, reduceOp);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            }
            break;
        }
        case ZCCLDataType::ZCCL_DATA_TYPE_BFP16: {
            ReduceScatterKernel<bfloat16_t> op;
            if (smallFlag) {
                op.Init(input, output, gva, rank, rankSize, totalLength, totalLength, magic, fftsAddr, smallFlag, reduceOp);
                op.Process();
            } else {
                op.RunBigDataOp(input, output, gva, rank, rankSize, totalLength, magic, fftsAddr, reduceOp);
            }
        }
        default:
            return;
    }
}