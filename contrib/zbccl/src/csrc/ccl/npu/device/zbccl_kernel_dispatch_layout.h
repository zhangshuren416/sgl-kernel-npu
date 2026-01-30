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
#ifndef ZBCCL_KERNEL_DISPATCH_LAYOUT_H
#define ZBCCL_KERNEL_DISPATCH_LAYOUT_H

#include <climits>
#include "kernel_operator.h"
#include "zbccl_def.h"
#include "zbccl_kernel_utils.h"
#include "zbccl_kernel_comm_args.h"

namespace MoeDispatchLayout {
using namespace AscendC;
using namespace Moe;

template <typename T>
class DispatchLayout {
public:
    __aicore__ inline DispatchLayout(){};

    __aicore__ inline void Init(GM_ADDR topkIdx, uint32_t numTokens, uint32_t numExperts, uint32_t numTopk,
                                uint32_t numRanks, uint32_t rank, GM_ADDR numTokensPerRank, GM_ADDR numTokensPerExpert,
                                GM_ADDR isTokenInRank, GM_ADDR sendTokenIdx, GM_ADDR notifySendData, TPipe* pipe)
    {
        numTokens_ = numTokens;
        numRanks_ = numRanks;
        rankId_ = rank;
        numExperts_ = numExperts;
        numTopk_ = numTopk;
        pipe_ = pipe;

        blockIdx_ = GetBlockIdx();
        uint32_t maxAivNum = GetBlockNum();
        blockNum_ = numTokens_ <= maxAivNum ? numTokens_ : maxAivNum;
        if (blockIdx_ >= blockNum_) {
            return;
        }
        uint32_t temp = numTokens_ / blockNum_;
        uint32_t restNum = numTokens_ % blockNum_;
        tempTokens_ = temp;
        if (blockIdx_ < restNum) {
            tempTokens_++;
        }
        topkIdx32AlignIntLen_ = Ceil(tempTokens_ * numTopk_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
        numTokensPerRank32AlignIntLen_ = Ceil(numRanks_ * sizeof(T), UB_ALIGN) * UB_ALIGN;
        numTokensPerExpert32AlignIntLen_ = Ceil(numExperts_ * sizeof(T), UB_ALIGN) * UB_ALIGN;
        isTokenInRank32AlignIntLen_ = Ceil(tempTokens_ * numRanks_ * sizeof(T), UB_ALIGN) * UB_ALIGN;
        sendTokenIdx32AlignIntLen_ = Ceil(tempTokens_ * numExperts_ * sizeof(T), UB_ALIGN) * UB_ALIGN;

        int64_t topkIdxOffset;
        int64_t isTokenOffset;
        if (blockIdx_ < restNum) {
            topkIdxOffset = blockIdx_ * tempTokens_ * numTopk_ * sizeof(int64_t);
            isTokenOffset = blockIdx_ * tempTokens_ * numRanks_ * sizeof(T);
        } else {
            topkIdxOffset = (restNum + blockIdx_ * tempTokens_) * numTopk_ * sizeof(int64_t);
            isTokenOffset = (restNum + blockIdx_ * tempTokens_) * numRanks_ * sizeof(T);
        }
        topkIdxGM_.SetGlobalBuffer((__gm__ int64_t*)(topkIdx + topkIdxOffset));
        numTokensPerRankGM_.SetGlobalBuffer((__gm__ T*)numTokensPerRank);
        numTokensPerExpertGM_.SetGlobalBuffer((__gm__ T*)numTokensPerExpert);
        isTokenInRankGM_.SetGlobalBuffer((__gm__ T*)(isTokenInRank + isTokenOffset));
        sendTokenIdxGM_.SetGlobalBuffer((__gm__ T*)(sendTokenIdx + topkIdxOffset / 2));
        tempExpertGM_.SetGlobalBuffer((__gm__ T*)notifySendData);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= blockNum_) {
            SyncAll<true>();
            return;
        }
        pipe_->Reset();
        pipe_->InitBuffer(topkIdxBuf_, topkIdx32AlignIntLen_);
        pipe_->InitBuffer(numTokensPerRankBuf_, numTokensPerRank32AlignIntLen_);
        pipe_->InitBuffer(numTokensPerExpertBuf_, numTokensPerExpert32AlignIntLen_);
        pipe_->InitBuffer(isTokenInRankBuf_, isTokenInRank32AlignIntLen_);
        pipe_->InitBuffer(seenRankBuf_, numRanks_ * sizeof(T));
        pipe_->InitBuffer(sendTokenIdxBuf_, topkIdx32AlignIntLen_);

        LocalTensor<int64_t> topkIdxTensor = topkIdxBuf_.AllocTensor<int64_t>();
        const DataCopyExtParams dataCopyParams{1U, topkIdx32AlignIntLen_, 0U, 0U, 0U};
        const DataCopyPadExtParams<int64_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(topkIdxTensor, topkIdxGM_, dataCopyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        LocalTensor<T> numTokensPerRankTensor = numTokensPerRankBuf_.AllocTensor<T>();
        LocalTensor<T> numTokensPerExpertTensor = numTokensPerExpertBuf_.AllocTensor<T>();
        LocalTensor<T> isTokenInRankTensor = isTokenInRankBuf_.AllocTensor<T>();
        LocalTensor<T> seenRankTensor = seenRankBuf_.AllocTensor<T>();
        LocalTensor<T> sendTokenIdxTensor = sendTokenIdxBuf_.AllocTensor<T>();
        Duplicate<T>(numTokensPerRankTensor, 0, numRanks_);
        Duplicate<T>(numTokensPerExpertTensor, 0, numTokensPerExpert32AlignIntLen_ / sizeof(T));
        Duplicate<T>(isTokenInRankTensor, 0, tempTokens_ * numRanks_);
        SyncFunc<AscendC::HardEvent::V_S>();

        int expertPerRank = numExperts_ / numRanks_;
        for (int i = 0; i < tempTokens_; ++i) {
            SyncFunc<AscendC::HardEvent::S_V>();
            Duplicate<T>(seenRankTensor, 0, numRanks_);
            SyncFunc<AscendC::HardEvent::V_S>();
            for (int j = 0; j < numTopk_; ++j) {
                int64_t expertId = topkIdxTensor.GetValue(i * numTopk_ + j);
                if (expertId < 0 || expertId >= numExperts_) {
                    continue;
                }
                uint32_t perExpertNum = numTokensPerExpertTensor.GetValue(expertId) + 1;
                numTokensPerExpertTensor.SetValue(expertId, perExpertNum);
                int rankId = expertId / expertPerRank;
                if (!seenRankTensor.GetValue(rankId)) {
                    uint32_t perRankNum = numTokensPerRankTensor.GetValue(rankId) + 1;
                    isTokenInRankTensor.SetValue(i * numRanks_ + rankId, 1);
                    seenRankTensor.SetValue(rankId, 1);
                    numTokensPerRankTensor.SetValue(rankId, perRankNum);
                }
            }
        }

        uint32_t sendSize = tempTokens_ * numRanks_ * sizeof(T);
        const DataCopyExtParams isTokenInRankDataCopyParams{1U, sendSize, 0U, 0U, 0U};
        DataCopyPad(isTokenInRankGM_, isTokenInRankTensor, isTokenInRankDataCopyParams);

        AscendC::SetAtomicAdd<T>();
        const DataCopyExtParams tempExpertDataCopyParams{1U, numTokensPerExpert32AlignIntLen_, 0U, 0U, 0U};
        for (int i = blockIdx_ + 1; i < blockNum_; ++i) {
            DataCopyPad(tempExpertGM_[i * numExperts_], numTokensPerExpertTensor, tempExpertDataCopyParams);
        }
        sendSize = numRanks_ * sizeof(T);
        const DataCopyExtParams numTokensPerRankDataCopyParams{1U, sendSize, 0U, 0U, 0U};
        DataCopyPad(numTokensPerRankGM_, numTokensPerRankTensor, numTokensPerRankDataCopyParams);
        sendSize = numExperts_ * sizeof(T);
        const DataCopyExtParams numTokensPerExpertDataCopyParams{1U, sendSize, 0U, 0U, 0U};
        DataCopyPad(numTokensPerExpertGM_, numTokensPerExpertTensor, numTokensPerExpertDataCopyParams);
        AscendC::SetAtomicNone();
        PipeBarrier<PIPE_MTE3>();
        SyncAll<true>();

        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        const DataCopyPadExtParams<T> tempPadParams{false, 0U, 0U, 0U};
        DataCopyPad(numTokensPerExpertTensor, tempExpertGM_[blockIdx_ * numExperts_], tempExpertDataCopyParams,
                    tempPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (int i = 0; i < tempTokens_; ++i) {
            for (int j = 0; j < numTopk_; ++j) {
                int64_t expertId = topkIdxTensor.GetValue(i * numTopk_ + j);
                if (expertId < 0 || expertId >= numExperts_) {
                    continue;
                }
                T valT = numTokensPerExpertTensor(expertId);
                sendTokenIdxTensor(i * numTopk_ + j) = valT;
                numTokensPerExpertTensor(expertId) = valT + 1;
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        const DataCopyExtParams sendTokenIdxCopyParams{1U, static_cast<uint32_t>(tempTokens_ * numTopk_ * sizeof(T)),
                                                       0U, 0U, 0U};
        DataCopyPad(sendTokenIdxGM_, sendTokenIdxTensor, sendTokenIdxCopyParams);
    }

private:
    TPipe* pipe_{nullptr};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{0};

    GlobalTensor<int64_t> topkIdxGM_;
    GlobalTensor<T> numTokensPerRankGM_;
    GlobalTensor<T> numTokensPerExpertGM_;
    GlobalTensor<T> isTokenInRankGM_;
    GlobalTensor<T> tempExpertGM_;
    GlobalTensor<T> sendTokenIdxGM_;

    TBuf<> topkIdxBuf_;
    TBuf<> numTokensPerRankBuf_;
    TBuf<> numTokensPerExpertBuf_;
    TBuf<> isTokenInRankBuf_;
    TBuf<> seenRankBuf_;
    TBuf<> sendTokenIdxBuf_;

    uint32_t numTokens_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    uint32_t numExperts_{0};
    uint32_t numTopk_{0};
    uint32_t tempTokens_{0};

    uint32_t topkIdx32AlignIntLen_{0};
    uint32_t numTokensPerRank32AlignIntLen_{0};
    uint32_t numTokensPerExpert32AlignIntLen_{0};
    uint32_t isTokenInRank32AlignIntLen_{0};
    uint32_t sendTokenIdx32AlignIntLen_{0};
};
}  // namespace MoeDispatchLayout

#endif  // ZBCCL_KERNEL_DISPATCH_LAYOUT_H