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
#ifndef ZBCCL_SMA_H
#define ZBCCL_SMA_H

#include "zbccl_common_includes.h"

namespace zbccl {
namespace sma {

static const char* kPytorchNPUAllocConf = "PYTORCH_NPU_ALLOC_CONF";
static const char* kMaxSplitSizeMB = "max_split_size_mb";
static const char* kGarbageCollectionThreshold = "garbage_collection_threshold";
// static const char* kExpandableSegments = "expandable_segments";
static const char* kBaseAddrAlignedKB = "base_addr_aligned_kb";
static const char* kPageSize = "page_size";
static const char* kSegmentSizeMB = "segment_size_mb";

static constexpr size_t kAlignRoundLarge = 16384;            // round up large allocs to 16 KB
static constexpr size_t kSmallBuffer = 2097152;              // "small" allocations are packed in 2 MiB blocks
static constexpr size_t kLargeBuffer = 20971520;             // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMB = 1024 * 1024;                          // 1 MB

class SecondaryMemoryAllocator : public ZReferable
{
public:
    virtual ~SecondaryMemoryAllocator() = default;

    /**
     * @brief Initialize the allocator
     *
     * @param options      [in] options for the allocator
     * @param flags        [in] extra flags
     * @return 0 is successful
     */
    virtual ZResult Initialize(zbccl_allocator_options *options, int32_t flags) noexcept = 0;

    /**
     * @brief Un-initialize the allocator
     *
     * @param flags        [in] extra flags
     */
    virtual void UnInitialize(int32_t flags) noexcept = 0;

    /**
     * @brief Allocate memory
     *
     * @param size         [in] size to be allocated
     * @param device       [in] device id
     * @param stream       [in] stream
     * @param flags        [in] optional flags
     * @param out          [out] pointer that allocated
     * @return 0 if successful
     */
    virtual ZResult Allocate(ssize_t size, int32_t device, aclrtStream stream, int32_t flags, void *&out) noexcept = 0;

    /**
     * @brief Free memory
     *
     * @param ptr          [in] memory pointer allocated by <i>Allocate</i>
     * @param size         [in] size of the memory
     * @param device       [in] device id
     * @param stream       [in] stream
     * @param flags        [in] optional flags
     * @return 0 if successful
     */
    virtual ZResult Free(void *ptr, ssize_t size, int32_t device, aclrtStream stream, int32_t flags) noexcept = 0;
};
using SMAPtr = ZRef<SecondaryMemoryAllocator>;
}  // namespace sma
}  // namespace zbccl

#endif  // ZBCCL_SMA_H
