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
#ifndef ZBCCL_KERNEL_COMM_ARGS_H
#define ZBCCL_KERNEL_COMM_ARGS_H

namespace Moe {
constexpr int MAX_RANK_SIZE = 384;
constexpr int PING_PONG_SIZE = 2;
constexpr int UB_ALIGN = 32;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t ADDR_OFFSET = 32U;

enum Op : int { COPYONLY = -1, ADD = 0, MUL = 1, MAX = 2, MIN = 3 };
enum MetaType : int { STATE = 0, ADDR = 1, FLAG = 2 };

// meta:  |-- sizeForCommGroupInfo --|-- sizeForParam --|-- sizeForExchangeAddress --|
//                                                      |- state -|- addr -|- flag -|
// 1 MB meta space is reserved, and offset 950 KB is used to write the cleared synchronization flag.
constexpr uint64_t META_FLAG_OFFSET = 950 * 1024UL;

constexpr uint64_t CYCLE_TO_TIME = 50;  // cycle num is converted into a fixed base unit of time, set at 50
constexpr uint64_t TIMEOUT_DETECTION_THRESHOLD = 50000000UL;

constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t SMALL_DATA_SIZE = 1 * 1024 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr static int32_t UB_HEAD_OFFSET = 96;
constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + UB_ALIGN;
}  // namespace Moe

#endif // ZBCCL_KERNEL_UTILS_H