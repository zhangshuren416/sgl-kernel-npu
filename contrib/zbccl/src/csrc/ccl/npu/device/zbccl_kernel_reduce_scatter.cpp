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

#include <cstdint>
#include "kernel_operator.h"
#include "zbccl_def.h"
#include "zbccl_kernel_utils.h"
#include "zbccl_comm_host_device_struct.h"

constexpr uint32_t DATA_ADDR_SIZE = 8;
constexpr uint32_t DATA_ADDR_INTERVAL = 8;
constexpr uint32_t FLAG_INTERVAL = 16;


ZBCCL_KERNEL uint64_t GetDataAddr(__gm__ void *metaAddr, uint32_t rank, uint32_t groupSize)
{
    uint32_t dataAddrOffset = rank * DATA_ADDR_INTERVAL;
    __gm__ uint64_t* dataGmAddr = (__gm__ uint64_t*)metaAddr + dataAddrOffset;
    dcciCacheline((__gm__ uint8_t *)dataGmAddr);
    return *dataGmAddr;
}

ZBCCL_KERNEL void SetDataAddr(__gm__ void *metaAddr, uint64_t val, uint32_t rank, uint32_t groupSize)
{
    uint32_t dataAddrOffset = rank * DATA_ADDR_INTERVAL;
    __gm__ uint64_t* dataGmAddr = (__gm__ uint64_t*)metaAddr + dataAddrOffset;
    *dataGmAddr = val;
    dcciCacheline((__gm__ uint8_t *)dataGmAddr);
}

ZBCCL_KERNEL int32_t GetFlag(__gm__ void *metaAddr, uint32_t rank, uint32_t groupSize)
{
    uint32_t dataAddrLength = groupSize * DATA_ADDR_SIZE * DATA_ADDR_INTERVAL;
    __gm__ int32_t* flagAddr = (__gm__ int32_t*)((__gm__ uint8_t*)metaAddr + dataAddrLength) + rank * FLAG_INTERVAL;
    dcciCacheline((__gm__ uint8_t *)flagAddr);
    return *flagAddr;
}

ZBCCL_KERNEL void SetFlag(__gm__ void *metaAddr, int32_t val, uint32_t rank, uint32_t groupSize)
{
    uint32_t dataAddrLength = groupSize * DATA_ADDR_SIZE * DATA_ADDR_INTERVAL;
    __gm__ int32_t* flagAddr = (__gm__ int32_t*)((__gm__ uint8_t*)metaAddr + dataAddrLength) + rank * FLAG_INTERVAL;
    *flagAddr = val;
    dcciCacheline((__gm__ uint8_t *)flagAddr);
}

ZBCCL_KERNEL void InitDataAddrAndFlag(__gm__ void *metaAddr, __gm__ void *inputAddr, uint32_t aivIndex,
                                      uint32_t rank, uint32_t groupSize, __gm__ uint64_t *counterAddress,
                                      __gm__ uint64_t *barrierAddress, uint64_t localDeviceMemSize,
                                      __gm__ uint16_t *peerGroupRank2WorldRank)
{
    if (aivIndex < groupSize) {
        SetFlag(metaAddr, 0, aivIndex, groupSize);
    }
    // last param useless.
    zbccl_barrier_all(rank, groupSize, localDeviceMemSize, counterAddress, barrierAddress, peerGroupRank2WorldRank);
    if (aivIndex < groupSize) {
        uint64_t dataAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(inputAddr));
        SetDataAddr(zbccl_ptr(metaAddr, rank, aivIndex, localDeviceMemSize, peerGroupRank2WorldRank), dataAddr, rank, groupSize);
        SetFlag(zbccl_ptr(metaAddr, rank, aivIndex, localDeviceMemSize, peerGroupRank2WorldRank), 1, rank, groupSize);
    }
}

template <typename T>
class ZeroBuffReduceScatterKernel
{
public:
    ZBCCL_KERNEL ZeroBuffReduceScatterKernel() {}

    ZBCCL_KERNEL void Init(GM_ADDR x, GM_ADDR y, GM_ADDR metaAddr, AscendC::TPipe *pipe,
                                uint32_t rank, uint32_t groupSize, uint32_t totalLength, uint32_t magic,
                                uint32_t atomicOp)
    {
        this->atomicOp = atomicOp;
        this->magic = magic;
        this->rank = rank;
        this->groupSize = groupSize;
        auto groupInfo = reinterpret_cast<__gm__ CommGroupInfo *>(metaAddr);
        this->groupInfo = groupInfo;
        __gm__ void *exchangeAddr = (__gm__ void *)(groupInfo->myAddressExchangeGva);

        const uint32_t aivNum = AscendC::GetBlockNum();
        const uint32_t aivIndex = AscendC::GetBlockIdx();

        uint32_t coreGroupNum = aivNum;
        uint32_t lenPerRank = totalLength;

        corePerRank = coreGroupNum / groupSize;
        coreRankIdx = aivIndex % corePerRank;
        coreTargetRank = aivIndex / corePerRank;

        InitDataAddrAndFlag(exchangeAddr, (__gm__ void *)x, aivIndex, rank, groupSize,
                            (__gm__ uint64_t *)&groupInfo->counter, (__gm__ uint64_t *)&groupInfo->barrier,
                            groupInfo->localDeviceMemSize, (__gm__ uint16_t *)&groupInfo->peerGroupRank2WorldRank);
        int32_t addrReadyFlag;
        do {
            addrReadyFlag = GetFlag((__gm__ void*)exchangeAddr, coreTargetRank, groupSize);
        } while (addrReadyFlag != 1);

        uint64_t inputAddr = GetDataAddr((__gm__ void*)exchangeAddr, coreTargetRank, groupSize);
        GM_ADDR inputPtr = (GM_ADDR)inputAddr;

        uint32_t lenPerRankAlignToCore = CeilDiv(lenPerRank, corePerRank) * corePerRank;
        uint32_t formerLength = lenPerRankAlignToCore / corePerRank;
        uint32_t tailLength = lenPerRank / corePerRank;
        uint32_t formerNum = lenPerRank % corePerRank;
        uint32_t tailNum = corePerRank - formerNum;
        uint32_t xOffset;
        uint32_t yOffset;

        if (coreRankIdx < formerNum) {
            lenPerCore = formerLength;
            xOffset = rank * lenPerRank + coreRankIdx * formerLength;
            yOffset = coreRankIdx * formerLength;
        } else {
            lenPerCore = tailLength;
            xOffset = rank * lenPerRank +
                formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
            yOffset = formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
        }

        xGm.SetGlobalBuffer((__gm__ T *)inputPtr + xOffset, lenPerCore);
        yGm.SetGlobalBuffer((__gm__ T *)y + yOffset, lenPerCore);
        if (lenPerCore * sizeof(T) > UB_DMA_MAX_SIZE) {
            pipe->InitBuffer(bindQueue, 1, UB_DMA_MAX_SIZE);
        } else {
            pipe->InitBuffer(bindQueue, 1, lenPerCore * sizeof(T));
        }
    }

    ZBCCL_KERNEL void Process()
    {
#ifdef __DAV_C220_VEC__

        uint32_t leftCopySize = lenPerCore * sizeof(T);
        AscendC::DataCopyPadExtParams<T> padParams;
        SetAtomicOp<T>(atomicOp);
        uint32_t times = 0;
        uint32_t preCopyNum = UB_DMA_MAX_SIZE / sizeof(T);

        do {
            uint32_t curCopySize = (leftCopySize > UB_DMA_MAX_SIZE) ? UB_DMA_MAX_SIZE : leftCopySize;
            AscendC::LocalTensor<T> xLocal = bindQueue.AllocTensor<T>();
            AscendC::DataCopyExtParams dataCopyParams(1, curCopySize, 0, 0, 0);
            AscendC::DataCopyPad(xLocal, xGm[times * preCopyNum], dataCopyParams, padParams);
            bindQueue.EnQue(xLocal);
            xLocal = bindQueue.DeQue<T>();
            AscendC::DataCopyPad(yGm[times * preCopyNum], xLocal, dataCopyParams);
            bindQueue.FreeTensor(xLocal);
            leftCopySize = (leftCopySize > UB_DMA_MAX_SIZE) ? leftCopySize - UB_DMA_MAX_SIZE : 0;
            times++;
        } while (leftCopySize > 0);

        AscendC::SetAtomicNone();
        // Sync Ensure Corresponding Tasks Done.
        // last param useless.
        zbccl_barrier_all(rank, groupSize, groupInfo->localDeviceMemSize, (__gm__ uint64_t *)&groupInfo->counter,
                          (__gm__ uint64_t *)&groupInfo->barrier, (__gm__ uint16_t *)&groupInfo->peerGroupRank2WorldRank);
#endif
    }

private:
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECIN, 1> bindQueue;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t rank;
    uint32_t atomicOp;
    uint32_t lenPerCore;
    uint32_t coreTargetRank;
    uint32_t coreRankIdx;
    uint32_t corePerRank;
    uint32_t magic;
    __gm__ CommGroupInfo *groupInfo;
    uint32_t groupSize;
};

extern "C" __global__ __aicore__ void ZeroBuffReduceScatter(
    GM_ADDR input, GM_ADDR output, GM_ADDR gva,
    uint64_t fftsAddr, uint32_t dataType, uint32_t totalLength,
    uint32_t rank, uint32_t groupSize, uint32_t reduceOp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::SetSyncBaseAddr(fftsAddr);
    uint32_t magic = 1;
    AscendC::TPipe pipe;
    zbccl_datatype_t zbcclDataType = static_cast<zbccl_datatype_t>(dataType);
    switch (zbcclDataType) {
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT8: {
            ZeroBuffReduceScatterKernel<int8_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT16: {
            ZeroBuffReduceScatterKernel<int16_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT32: {
            ZeroBuffReduceScatterKernel<int32_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP32: {
            ZeroBuffReduceScatterKernel<float> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP16: {
            ZeroBuffReduceScatterKernel<float16_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_BFP16: {
            ZeroBuffReduceScatterKernel<bfloat16_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        default:
            return;
    }
}

int32_t ZBCCLOpReduceScatter(const void *inp, void *out, size_t recvNumel, zbccl_datatype_t dataType,
                           aclrtStream stream, zbccl_reduce_op_t reduceOp, const CommGroupInfo &groupInfo)
{
    /* define the block dim */
    uint32_t blockDim = 16;
    uint32_t dataTypeNum = static_cast<uint32_t>(dataType);
    uint32_t reduceOpNum = static_cast<uint32_t>(reduceOp);

    // Prepare FFTS address
    uint64_t fftsAddr = groupInfo.fftsConfig;
    uint16_t rank = groupInfo.myGroupRank;
    uint16_t groupSize = groupInfo.groupSize;
    uint8_t* metaAddr = reinterpret_cast<uint8_t *>(groupInfo.myMetaGva);
    uint8_t* input = reinterpret_cast<uint8_t *>(const_cast<void *>(inp));
    uint8_t* output = reinterpret_cast<uint8_t *>(out);

    ZeroBuffReduceScatter<<<blockDim, nullptr, stream>>>(input, output, metaAddr, fftsAddr, dataTypeNum, recvNumel, rank, groupSize, reduceOpNum);

    return 0;
}
