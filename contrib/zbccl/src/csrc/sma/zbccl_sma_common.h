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
#ifndef ZBCCL_SMA_COMMON_H
#define ZBCCL_SMA_COMMON_H

#include <set>
#include <cstdint>
#include <memory>
#include <c10/core/Allocator.h>

#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h>
#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>

#include "zbccl_common_includes.h"

namespace zbccl {
namespace sma {
/**
* @brief Type of block type
*/
enum DeviceBlockType {
    BT_SMALL,
    BT_BIG,
};

static const char* kPytorchNPUAllocConf = "PYTORCH_NPU_ALLOC_CONF";
static const char* kMaxSplitSizeMB = "max_split_size_mb";
static const char* kGarbageCollectionThreshold = "garbage_collection_threshold";
// static const char* kExpandableSegments = "expandable_segments";
// static const char* kBaseAddrAlignedKB = "base_addr_aligned_kb";
static const char* kPageSize = "page_size";
static const char* kSegmentSizeMB = "segment_size_mb";

constexpr size_t kMinBlockSize = 512;                 // all sizes are rounded to at least 512 bytes(for L1)
constexpr size_t kSmallSize = 1048576;                // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;              // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;             // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kSmallAlloc = kSmallSize;            // allocations under 1 MiB may use kSmallBuffer
constexpr size_t kMiddleAlloc = 10485760;             // allocations between 1 and 10 MiB may use kLargeBuffer
                                                      // allocations over 10MiB using rounded size with kRoundLarge
constexpr size_t kRoundLarge = 2097152;               // round up large allocs to 2 MiB
// constexpr size_t kAlignRoundLarge = 16384;            // align large allocs head addr to 16 KB
constexpr size_t kMB = 1024 * 1024;                   // 1 MB
constexpr size_t kSmallHeapSize = 512 * kMB;          // 512MB for small heap in dualHeap allocator

}  // namespace sma
}  // namespace zbccl

#endif  // ZBCCL_SMA_COMMON_H
