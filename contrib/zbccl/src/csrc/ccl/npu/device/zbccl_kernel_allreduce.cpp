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
                                      __gm__ uint16_t *peerGroupRank2WorldRank, __gm__ void *paramAddr)
{
    if (aivIndex < groupSize) {
        SetFlag(metaAddr, 0, aivIndex, groupSize);
    }
    Barrier(paramAddr, rank, groupSize, localDeviceMemSize, peerGroupRank2WorldRank);
    if (aivIndex < groupSize) {
        uint64_t dataAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(inputAddr));
        SetDataAddr(zbccl_ptr(metaAddr, rank, aivIndex, localDeviceMemSize, peerGroupRank2WorldRank), dataAddr, rank, groupSize);
        SetFlag(zbccl_ptr(metaAddr, rank, aivIndex, localDeviceMemSize, peerGroupRank2WorldRank), 1, rank, groupSize);
    }
}

ZBCCL_KERNEL void zbccl_barrier_all_for_ar(uint16_t rankId, uint16_t groupSize, uint64_t localSize,
                                    __gm__ uint64_t *counterAddress, __gm__ uint64_t *barrierAddress,
                                    __gm__ uint16_t *peerRanks)
{
    int vecId = AscendC::GetBlockIdx();
    int vecSize = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    uint64_t count = zbccl_load<uint64_t>(counterAddress) + 1;
    int k = 8;
    k = k < groupSize ? k : groupSize;
    k = k < vecSize ? k : vecSize;

    AscendC::SyncAll<true>();

    if ASCEND_IS_AIV {
        if (vecId == rankId) {
            zbccl_single_set(barrierAddress, count);
        }
        for (int i = vecId; i < groupSize; i += k) {
            __gm__ void *dst = zbccl_ptr((__gm__ void *)barrierAddress, rankId, i, localSize, peerRanks);
            __gm__ uint64_t *target_addr = (__gm__ uint64_t *)dst;
            zbccl_single_wait_until_eq(target_addr, count);
        }
        if (vecId == rankId) {
            zbccl_single_set(counterAddress, count);
        }
    }

    AscendC::SyncAll<true>();
}

template <typename T>
class ZeroBuffAllReduceKernel
{
public:
    ZBCCL_KERNEL ZeroBuffAllReduceKernel() {}

    ZBCCL_KERNEL void Init(GM_ADDR x, GM_ADDR y, GM_ADDR metaAddr, GM_ADDR buf, AscendC::TPipe *pipe,
                                uint32_t rank, uint32_t groupSize, uint32_t totalLength, uint32_t bufLength, uint32_t magic,
                                uint32_t atomicOp)
    {
        this->atomicOp = atomicOp;
        this->magic = magic;
        this->rank = rank;
        this->groupSize = groupSize;
        this->bufLen = bufLength / sizeof(T);
        this->totalLen = totalLength;
        auto groupInfo = reinterpret_cast<__gm__ CommGroupInfo *>(metaAddr);
        this->groupInfo = groupInfo;
        this->pipe = pipe;
        __gm__ void *exchangeAddr = (__gm__ void *)(groupInfo->myAddressExchangeGva);
        __gm__ void *paramAddr = (__gm__ void *)(groupInfo->myParamDataGva);

        const uint32_t aivNum = AscendC::GetBlockNum();
        const uint32_t aivIndex = AscendC::GetBlockIdx();

        corePerRank = aivNum / groupSize;
        coreRankIdx = aivIndex % corePerRank;
        coreTargetRank = aivIndex / corePerRank;

        InitDataAddrAndFlag(exchangeAddr, (__gm__ void *)x, aivIndex, rank, groupSize,
                            reinterpret_cast<__gm__ uint64_t *>(groupInfo->vecCounter),
                            reinterpret_cast<__gm__ uint64_t *>(groupInfo->vecBarrier),
                            groupInfo->localDeviceMemSize, (__gm__ uint16_t *)groupInfo->peerGroupRank2WorldRank,
                            paramAddr);
        int32_t addrReadyFlag;
        do {
            addrReadyFlag = GetFlag((__gm__ void*)exchangeAddr, coreTargetRank, groupSize);
        } while (addrReadyFlag != 1);

        uint64_t inputAddr = GetDataAddr((__gm__ void*)exchangeAddr, coreTargetRank, groupSize);
        GM_ADDR inputPtr = (GM_ADDR)inputAddr;
        xGm.SetGlobalBuffer((__gm__ T *)inputPtr, totalLength);
        yGm.SetGlobalBuffer((__gm__ T *)y, totalLength);
        buffGm.SetGlobalBuffer((__gm__ T *)buf, bufLen);
    }

    ZBCCL_KERNEL void DataCopyGM2GM(AscendC::GlobalTensor<T> inputTensor,
                                    AscendC::GlobalTensor<T> outputTensor,
                                    uint32_t len, uint32_t inputOffset, uint32_t outputOffset)
    {
        AscendC::DataCopyPadExtParams<T> padParams;
        uint32_t leftCopySize = len * sizeof(T);
        uint32_t times = 0;
        uint32_t preCopyNum = UB_DMA_MAX_SIZE / sizeof(T);
        do {
            uint32_t curCopySize = (leftCopySize > UB_DMA_MAX_SIZE) ? UB_DMA_MAX_SIZE : leftCopySize;
            AscendC::LocalTensor<T> xLocal = bindQueue.AllocTensor<T>();
            AscendC::DataCopyExtParams dataCopyParams(1, curCopySize, 0, 0, 0);
            AscendC::DataCopyPad(xLocal, inputTensor[inputOffset + times * preCopyNum], dataCopyParams, padParams);
            bindQueue.EnQue(xLocal);
            xLocal = bindQueue.DeQue<T>();
            AscendC::DataCopyPad(outputTensor[outputOffset + times * preCopyNum], xLocal, dataCopyParams);
            bindQueue.FreeTensor(xLocal);
            leftCopySize = (leftCopySize > UB_DMA_MAX_SIZE) ? leftCopySize - UB_DMA_MAX_SIZE : 0;
            times++;
        } while (leftCopySize > 0);
    }

    ZBCCL_KERNEL void Process()
    {
#ifdef __DAV_C220_VEC__
        uint32_t bufLoopTimes = CeilDiv(totalLen, bufLen);
        uint32_t tailLen = (bufLoopTimes == 1) ? totalLen : totalLen % bufLen;
        for (uint32_t i = 0; i < bufLoopTimes; ++i) {
            // 计算输入数据在每个核的偏移
            uint32_t lenPerRank = (i == bufLoopTimes - 1) ? tailLen : bufLen;
            uint32_t lenPerRankAlignToCore = CeilDiv(lenPerRank, corePerRank) * corePerRank;
            uint32_t formerLength = lenPerRankAlignToCore / corePerRank;
            uint32_t tailLength = lenPerRank / corePerRank;
            uint32_t formerNum = lenPerRank % corePerRank;
            uint32_t tailNum = corePerRank - formerNum;
            uint32_t loopOffset = i * bufLen;
            uint32_t coreOffset;

            if (coreRankIdx < formerNum) {
                lenPerCore = formerLength;
                coreOffset = coreRankIdx * formerLength;
            } else {
                lenPerCore = tailLength;
                coreOffset = formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
            }
            
            if (lenPerCore * sizeof(T) > UB_DMA_MAX_SIZE) {
                pipe->InitBuffer(bindQueue, 1, UB_DMA_MAX_SIZE);
            } else {
                pipe->InitBuffer(bindQueue, 1, lenPerCore * sizeof(T));
            }

            // 搬运数据input -> buffer
            uint32_t inOffset = loopOffset + coreOffset;
            uint32_t outOffset = coreOffset;

            if (rank == coreTargetRank) {
                AscendC::SetAtomicNone();
                DataCopyGM2GM(xGm, buffGm, lenPerCore, inOffset, outOffset);
                AscendC::SyncAll<true>();
            } else {
                AscendC::SyncAll<true>();
                SetAtomicOp<T>(atomicOp);
                DataCopyGM2GM(xGm, buffGm, lenPerCore, inOffset, outOffset);
                AscendC::SetAtomicNone();
            }

            Barrier((__gm__ void *)(groupInfo->myParamDataGva), rank, groupSize, groupInfo->localDeviceMemSize,
                    (__gm__ uint16_t *)groupInfo->peerGroupRank2WorldRank);

            // 搬运数据buffer -> output
            const uint32_t aivNum = AscendC::GetBlockNum();
            const uint32_t aivIndex = AscendC::GetBlockIdx();
            lenPerRankAlignToCore = CeilDiv(lenPerRank, aivNum) * aivNum;
            formerLength = lenPerRankAlignToCore / aivNum;
            tailLength = lenPerRank / aivNum;
            formerNum = lenPerRank % aivNum;
            tailNum = aivNum - formerNum;
            if (aivIndex < formerNum) {
                lenPerCore = formerLength;
                coreOffset = aivIndex * formerLength;
            } else {
                lenPerCore = tailLength;
                coreOffset = formerNum * formerLength + (aivIndex - formerNum) * tailLength;
            }

            pipe->Reset();
            if (lenPerCore * sizeof(T) > UB_DMA_MAX_SIZE) {
                pipe->InitBuffer(bindQueue, 1, UB_DMA_MAX_SIZE);
            } else {
                pipe->InitBuffer(bindQueue, 1, lenPerCore * sizeof(T));
            }

            inOffset = coreOffset;
            outOffset = loopOffset + coreOffset;
            DataCopyGM2GM(buffGm, yGm, lenPerCore, inOffset, outOffset);
            pipe->Reset();
        }
#endif
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> bindQueue;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<T> buffGm;
    uint32_t rank;
    uint32_t atomicOp;
    uint32_t lenPerCore;
    uint32_t coreTargetRank;
    uint32_t coreRankIdx;
    uint32_t corePerRank;
    uint32_t magic;
    uint32_t groupSize;
    uint32_t bufLen;
    uint32_t totalLen;
    __gm__ CommGroupInfo *groupInfo;
};

extern "C" __global__ __aicore__ void ZeroBuffAllReduce(
    GM_ADDR input, GM_ADDR output, GM_ADDR buffer, GM_ADDR gva,
    uint64_t fftsAddr, uint32_t dataType, uint32_t totalLength, uint32_t bufLen,
    uint32_t rank, uint32_t groupSize, uint32_t reduceOp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::SetSyncBaseAddr(fftsAddr);
    uint32_t magic = 1;
    AscendC::TPipe pipe;
    zbccl_datatype_t zbcclDataType = static_cast<zbccl_datatype_t>(dataType);
    switch (zbcclDataType) {
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT8: {
            ZeroBuffAllReduceKernel<int8_t> op;
            op.Init(input, output, gva, buffer, &pipe, rank, groupSize, totalLength, bufLen, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT16: {
            ZeroBuffAllReduceKernel<int16_t> op;
            op.Init(input, output, gva, buffer, &pipe, rank, groupSize, totalLength, bufLen, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT32: {
            ZeroBuffAllReduceKernel<int32_t> op;
            op.Init(input, output, gva, buffer, &pipe, rank, groupSize, totalLength, bufLen, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP32: {
            ZeroBuffAllReduceKernel<float> op;
            op.Init(input, output, gva, buffer, &pipe, rank, groupSize, totalLength, bufLen, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP16: {
            ZeroBuffAllReduceKernel<float16_t> op;
            op.Init(input, output, gva, buffer, &pipe, rank, groupSize, totalLength, bufLen, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_BFP16: {
            ZeroBuffAllReduceKernel<bfloat16_t> op;
            op.Init(input, output, gva, buffer, &pipe, rank, groupSize, totalLength, bufLen, magic, reduceOp);
            op.Process();
            break;
        }
        default:
            return;
    }
}

int32_t ZBCCLOpAllReduce(const void *inp, void *out, void *buf, size_t numel, size_t buf_cnt, zbccl_datatype_t dataType,
                           aclrtStream stream, zbccl_reduce_op_t reduceOp, const CommGroupInfo &groupInfo)
{
    /* define the block dim */
    uint32_t blockDim = 32;
    uint32_t dataTypeNum = static_cast<uint32_t>(dataType);
    uint32_t reduceOpNum = static_cast<uint32_t>(reduceOp);

    // Prepare FFTS address
    uint64_t fftsAddr = groupInfo.fftsConfig;
    uint16_t rank = groupInfo.myGroupRank;
    uint16_t groupSize = groupInfo.groupSize;
    uint8_t* metaAddr = reinterpret_cast<uint8_t *>(groupInfo.myMetaGva);
    uint8_t* input = reinterpret_cast<uint8_t *>(const_cast<void *>(inp));
    uint8_t* output = reinterpret_cast<uint8_t *>(out);
    uint8_t *buffer = reinterpret_cast<uint8_t *>(buf);

    ZeroBuffAllReduce<<<blockDim, nullptr, stream>>>(input, output, buffer, metaAddr, fftsAddr, dataTypeNum, numel, buf_cnt, rank, groupSize, reduceOpNum);

    return 0;
}
