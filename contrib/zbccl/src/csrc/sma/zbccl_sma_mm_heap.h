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
#include "zbccl_sma_common.h"
#include "zbccl_sma_config.h"

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

    size_t getTotalSize() noexcept;

    size_t getInUsedSize() noexcept;

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
    uint64_t used_size_;
    mutable pthread_spinlock_t spinlock_{};
    std::map<uint64_t, uint64_t> address_idle_tree_;
    std::map<uint64_t, uint64_t> address_used_tree_;
    std::set<MemoryRange, RangeSizeFirstComparator> size_idle_tree_;
};

class CustomMemoryHeap {
public:
    CustomMemoryHeap(void *base, uint64_t size) : base_(static_cast<uint8_t*>(base)), size_(size) {};

    virtual ~CustomMemoryHeap() noexcept = default;

public:
    virtual void *alignedAllocate(uint64_t alignment, uint64_t size) noexcept = 0;

    virtual size_t getTotalSize() noexcept = 0;

    virtual size_t getInUsedSize() noexcept = 0;

    virtual int32_t release(void *address) noexcept = 0;

    virtual bool allocatedSize(void *address, uint64_t &size) const noexcept = 0;

    inline bool isInitialized() const noexcept { return initialized_; }

    // to be deprecated
    inline uint8_t *getBaseAddr() const noexcept { return base_; }

protected:
    bool initialized_{false};

    uint8_t *const base_;
    const uint64_t size_;
};

class DualMemoryHeap : public CustomMemoryHeap {
public:
    DualMemoryHeap(void *base, uint64_t size);

    ~DualMemoryHeap() noexcept override = default;

public:
    void *alignedAllocate(uint64_t alignment, uint64_t size) noexcept override;

    size_t getTotalSize() noexcept override;

    size_t getInUsedSize() noexcept override;

    int32_t release(void *address) noexcept override;

    bool allocatedSize(void *address, uint64_t &size) const noexcept override;

private:
    const uint64_t SMALL_SIZE{SMAConfig::small_heap_size()};       // small heap total size kSmallHeapSize
    const uint64_t SMALL_ALLOC{SMAConfig::small_heap_threshold()};     // small heap cover under kSmallThreshold alloc
    constexpr static bool ENABLE_CROSS{true};                   // whether enable small alloc overflow to large heap

    const uint64_t size_small_;
    const uint64_t size_large_;
    MemoryHeap small_;
    MemoryHeap large_;

};

// =========================================================
// SplitMemoryHeap Class Declaration
// Features: O(log N) Search, Bi-directional Allocation, POSIX Spinlock
// =========================================================
class SplitMemoryHeap : public CustomMemoryHeap {
public:
    SplitMemoryHeap(void *base, uint64_t size, uint64_t threshold = SMAConfig::small_heap_threshold());
    ~SplitMemoryHeap() override;

    // Core Interface Implementation
    void *alignedAllocate(uint64_t alignment, uint64_t size) noexcept override;
    int32_t release(void *address) noexcept override;
    bool allocatedSize(void *address, uint64_t &size) const noexcept override;
    size_t getTotalSize() noexcept override;
    size_t getInUsedSize() noexcept override;

private:
    // Configuration
    uint64_t split_threshold_;
    size_t used_bytes_{0};

    // Locking: POSIX Spinlock
    mutable pthread_spinlock_t spinlock_{};

    // --- RAII Helper for Spinlock ---
    // Defined inline for performance (compiler optimization)
    struct SpinGuard {
        pthread_spinlock_t& lock_ref;

        explicit SpinGuard(pthread_spinlock_t& lock) : lock_ref(lock) {
            pthread_spin_lock(&lock_ref);
        }

        ~SpinGuard() {
            pthread_spin_unlock(&lock_ref);
        }

        // Disable copy/move to prevent accidental unlocking
        SpinGuard(const SpinGuard&) = delete;
        SpinGuard& operator=(const SpinGuard&) = delete;
    };

    // --- Data Structures ---

    // 1. Global Address View: Key=Address, Value=Size
    // Used for O(log N) Coalescing (Merging)
    std::map<uintptr_t, size_t> global_free_map_;

    // 2. Segregated Size View: buckets[k] contains addresses of blocks with size in [2^k, 2^(k+1))
    // The internal std::set is sorted by Address.
    std::vector<std::set<uintptr_t>> size_buckets_;

    // 3. Metadata for allocated blocks
    std::unordered_map<void*, uint64_t> allocated_records_;

    // --- Internal Helper Methods ---

    static int get_bucket_index(uint64_t size);

    void addToFreeStructures(uintptr_t addr, uint64_t size);
    void removeFromFreeStructures(uintptr_t addr, uint64_t size);

    void* allocateLowToHigh(int start_idx, uint64_t alignment, uint64_t req_size);
    void* allocateHighToLow(int start_idx, uint64_t alignment, uint64_t req_size);

    void coalesceAndInsert(uintptr_t addr, uint64_t size);
};

}  // namespace heap

// Helper: Bit manipulation for bucket indexing
inline int get_bucket_index(uint64_t size) {
    if (size == 0) return 0;
    // Uses GCC/Clang built-in for O(1) calculation.
    return 64 - __builtin_clzll(size) - 1;
}

// Heap API remains for dma
ZBCCL_API int HeapAlignedAllocate(void **devPtr, size_t size,
                                  std::shared_ptr<heap::MemoryHeap> shmem_pool);

ZBCCL_API int HeapRelease(void *devPtr, std::shared_ptr<heap::MemoryHeap> shmem_pool);

ZBCCL_API int GetTotalSize(size_t &size, std::shared_ptr<heap::MemoryHeap> shmem_pool);

ZBCCL_API int GetInUsedSize(size_t &size, std::shared_ptr<heap::MemoryHeap> shmem_pool);

ZBCCL_API int CustomHeapAlignedAllocate(void **devPtr, size_t size,
                                        std::shared_ptr<heap::CustomMemoryHeap> shmem_pool);

ZBCCL_API int CustomHeapRelease(void *devPtr, std::shared_ptr<heap::CustomMemoryHeap> shmem_pool);

ZBCCL_API int CustomGetTotalSize(size_t &size, std::shared_ptr<heap::CustomMemoryHeap> shmem_pool);

ZBCCL_API int CustomInUsedSize(size_t &size, std::shared_ptr<heap::CustomMemoryHeap> shmem_pool);

}  // namespace sma
}  // namespace zbccl

#endif  // ZBCCL_SMA_MM_HEAP_H
