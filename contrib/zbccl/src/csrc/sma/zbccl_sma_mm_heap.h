/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#ifndef ZBCCL_SMA_MM_HEAP_H
#define ZBCCL_SMA_MM_HEAP_H

#include <pthread.h>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <iostream>

#include "zbccl_common_includes.h"

namespace zbccl {
namespace sma {
namespace heap {

struct MemoryRange {
    const uint64_t offset_;
    const uint64_t size_;

    MemoryRange(uint64_t o, uint64_t s) noexcept : offset_{o}, size_{s} {}
};

struct RangeSizeFirstComparator {
    bool operator()(const MemoryRange &mr1,
                    const MemoryRange &mr2) const noexcept;
};

class MemoryHeap {
public:
    MemoryHeap(void *base, uint64_t size) noexcept;

    ~MemoryHeap() noexcept;

public:
    void *allocate(uint64_t size) noexcept;

    void *alignedAllocate(uint64_t alignment, uint64_t size) noexcept;

    bool changeSize(void *address, uint64_t size) noexcept;

    size_t reservedTotalSize() noexcept;

    int32_t release(void *address) noexcept;

    bool allocatedSize(void *address, uint64_t &size) const noexcept;

private:
    static uint64_t allocated_size_align_up(uint64_t input_size) noexcept;

    static bool alignment_matches(const MemoryRange &mr, uint64_t alignment,
                                  uint64_t size, uint64_t &head_skip) noexcept;

    void reduce_size_in_lock(const std::map<uint64_t, uint64_t>::iterator &pos,
                             uint64_t new_size) noexcept;

    bool expend_size_in_lock(const std::map<uint64_t, uint64_t>::iterator &pos,
                             uint64_t new_size) noexcept;

private:
    bool initialized_{false};
    uint8_t *const base_;
    const uint64_t size_;
    mutable pthread_spinlock_t spinlock_{};
    std::map<uint64_t, uint64_t> address_idle_tree_;
    std::map<uint64_t, uint64_t> address_used_tree_;
    std::set<MemoryRange, RangeSizeFirstComparator> size_idle_tree_;
};

}  // namespace heap

ZBCCL_API int HeapAlignedAllocate(void **devPtr, size_t size,
                                  std::shared_ptr <heap::MemoryHeap> shmem_pool);

ZBCCL_API int HeapRelease(void *devPtr, std::shared_ptr <heap::MemoryHeap> shmem_pool);

ZBCCL_API int ReservedTotalSize(size_t &size, std::shared_ptr <heap::MemoryHeap> shmem_pool);

}  // namespace sma
}  // namespace zbccl

#endif  // ZBCCL_SMA_MM_HEAP_H
