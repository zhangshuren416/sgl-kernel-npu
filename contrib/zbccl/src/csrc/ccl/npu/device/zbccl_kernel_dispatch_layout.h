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
#include "zbccl_comm_host_device_struct.h"
#include "zbccl_kernel_barrier.h"

namespace MoeDispatchLayout {
using namespace AscendC;

constexpr uint32_t UB_32_ALIGN = 32U;

template <typename T>
class DispatchLayout
{
public:
    __aicore__ inline DispatchLayout(){};

    __aicore__ inline void Init(GM_ADDR topkIdx, uint32_t numTokens, uint32_t numExperts, uint32_t numTopk,
                                uint32_t numRanks, uint32_t rank, GM_ADDR numTokensPerRank, GM_ADDR numTokensPerExpert,
                                GM_ADDR isTokenInRank, GM_ADDR sendTokenIdx)
    {
    }

    __aicore__ inline void Process()
    {

    }

// private:
//     GlobalTensor<int64_t> topkIdxGM_;
//     GlobalTensor<T> numTokensPerRankGM_;
//     GlobalTensor<T> numTokensPerExpertGM_;
//     GlobalTensor<T> isTokenInRankGM_;
//     GlobalTensor<T> tempExpertGM_;
//     GlobalTensor<T> sendTokenIdxSmallGM_;

//     TBuf<> topkIdxBuf_;
//     TBuf<> numTokensPerRankBuf_;
//     TBuf<> numTokensPerExpertBuf_;
//     TBuf<> isTokenInRankBuf_;
//     TBuf<> seenRankBuf_;
//     TBuf<> sendTokenIdxSmallBuf_;

//     TPipe *tpipe_{nullptr};
//     uint32_t numTokens_{0};
//     uint32_t numRanks_{0};
//     uint32_t numExperts_{0};
//     uint32_t numTopk_{0};
//     uint32_t coreIdx_{0};
//     uint32_t aivNum_{0};
//     uint32_t tempTokens_{0};

//     uint32_t topkIdx32AlignIntLen_{0};
//     uint32_t numTokensPerRank32AlignIntLen_{0};
//     uint32_t numTokensPerExpert32AlignIntLen_{0};
//     uint32_t isTokenInRank32AlignIntLen_{0};
//     uint32_t sendTokenIdx32AlignIntLen_{0};
};
}  // namespace MoeDispatchLayout

#endif  // ZBCCL_KERNEL_DISPATCH_LAYOUT_H