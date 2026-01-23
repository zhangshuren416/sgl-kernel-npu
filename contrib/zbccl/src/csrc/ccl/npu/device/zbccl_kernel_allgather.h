/*

Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
ZBCCL is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
     http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
*/
#ifndef ZBCCL_KERNEL_ALLGATHER_H
#define ZBCCL_KERNEL_ALLGATHER_H
#include "kernel_operator.h"
#include "shmem_api.h"
#include "zbccl_def.h"
#include "zbccl_kernel_utils.h"


class AllGatherKernel
{
public:
    __aicore__ inline AllGatherKernel() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, GM_ADDR metaGM, uint64_t elements, 
        uint16_t groupSize, uint16_t myGroupRank)
    {
#ifdef DAV_C220_VEC
        shmem_barrier_all();
        uint64_t flagMagic = 1024;
        uint16_t groupSize = commMeta->groupSize;
        uint16_t myGroupRank = commMeta->myGroupRank;
        ExchangeInputAddr(inputGM, metaGM, groupSize, myGroupRank, flagMagic);

        const int64_t aivNum = AscendC::GetBlockNum();
        const int64_t aivIndex = AscendC::GetBlockIdx();
    
        // data move parameters
        const int64_t corePerRank = aivNum / groupSize; // 4 
        const int64_t coreRankIdx = aivIndex % corePerRank; // 0,1,2,3 
        const int64_t x = aivIndex / corePerRank; //0,1 

        uint64_t flag = 0;
        AscendC::LocalTensor<uint64_t> flagBuff(AscendC::TPosition::VECIN, 2*64 + 32, 1);
        flagBuff(0) = 0;
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);

        AscendC::GlobalTensor<uint64_t> meta_addr_tensor;
        meta_addr_tensor.SetGlobalBuffer((__gm__ uint64_t *)metaGM, groupSize * FLAG_SIZE * 2);
        AscendC::DataCopyPadExtParams<uint64_t> copyExtParams{false, 0U, 0U, 0U};
        AscendC::DataCopyExtParams copyParams{1U, static_cast<uint32_t>(64), 0, 0, 0};

        while (flag != flagMagic) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::DataCopyPad(flagBuff, meta_addr_tensor[groupSize * FLAG_SIZE + x * FLAG_SIZE], copyParams, copyExtParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            flag = flagBuff.GetValue(0);
        }

        AscendC::LocalTensor<uint64_t> input_addr_buff(AscendC::TPosition::VECIN, 3*64 + 32, 1);
        AscendC::GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer((__gm__ T *)outputGM, elements * groupSize);

        uint32_t numPerCore = elements / corePerRank; 
        uint32_t outputOffset = x * elements + coreRankIdx * numPerCore;
        uint32_t inputOffset = coreRankIdx * numPerCore;
        if (coreRankIdx == corePerRank - 1) {
            numPerCore = elements - (corePerRank - 1) * numPerCore;
        }

        // get input addr
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(input_addr_buff, meta_addr_tensor[x * FLAG_SIZE], copyParams, copyExtParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        AscendC::GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer((__gm__ T *)input_addr_buff.GetValue(0), elements);

        AscendC::PipeBarrier<PIPE_ALL>();
        CpGM2GM(outputGT[outputOffset], inputGT[inputOffset], numPerCore);
#endif
    }

};

#endif // ZBCCL_KERNEL_ALLGATHER_H