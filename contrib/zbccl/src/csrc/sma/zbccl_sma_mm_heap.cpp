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
#include "zbccl_defines.h"
#include "zbccl_sma_mm_heap.h"

constexpr int32_t ALIGN_32 = 32;

namespace zbccl {
namespace sma {
namespace heap {

bool RangeSizeFirstComparator::operator()(
        const MemoryRange &mr1, const MemoryRange &mr2) const noexcept {
    if (mr1.size_ != mr2.size_) {
        return mr1.size_ < mr2.size_;
    }

    return mr1.offset_ < mr2.offset_;
}

// MemoryHeap
MemoryHeap::MemoryHeap(void *base, uint64_t size) noexcept
        : base_{reinterpret_cast<uint8_t *>(base)}, size_{size}, used_size_{0} {
    pthread_spin_init(&spinlock_, 0);
    address_idle_tree_[0] = size;
    size_idle_tree_.insert({0, size});
}

MemoryHeap::~MemoryHeap() noexcept { pthread_spin_destroy(&spinlock_); }

void *MemoryHeap::allocate(uint64_t size) noexcept {
    if (size == 0 || size > size_) {
        ZBCCL_LOG_WARN("cannot allocate with size " << size);
        return nullptr;
    }

    auto aligned_size = allocated_size_align_up(size);
    MemoryRange anchor{0, aligned_size};

    pthread_spin_lock(&spinlock_);
    auto size_pos = size_idle_tree_.lower_bound(anchor);
    if (size_pos == size_idle_tree_.end()) {
        pthread_spin_unlock(&spinlock_);
        ZBCCL_LOG_WARN("cannot allocate with size: " << size);
        return nullptr;
    }

    auto target_offset = size_pos->offset_;
    auto target_size = size_pos->size_;
    auto addr_pos = address_idle_tree_.find(target_offset);
    if (addr_pos == address_idle_tree_.end()) {
        pthread_spin_unlock(&spinlock_);
        ZBCCL_LOG_ERROR("offset(" << target_offset << ") size(" << target_size <<
                                        ") in size tree, not in address tree.");
        return nullptr;
    }

    size_idle_tree_.erase(size_pos);
    address_idle_tree_.erase(addr_pos);
    address_used_tree_.emplace(target_offset, aligned_size);
    if (target_size > aligned_size) {
        MemoryRange left{target_offset + aligned_size, target_size - aligned_size};
        address_idle_tree_.emplace(left.offset_, left.size_);
        size_idle_tree_.emplace(left);
    }
    used_size_ += aligned_size;
    pthread_spin_unlock(&spinlock_);

    return base_ + target_offset;
}

void *MemoryHeap::alignedAllocate(uint64_t alignment, uint64_t size) noexcept {
    if (size == 0 || alignment == 0 || size > size_) {
        ZBCCL_LOG_ERROR("invalid input, align=" << alignment << ", size=" << size << ", total=" << size_);
        return nullptr;
    }

    if ((alignment & (alignment - 1UL)) != 0) {
        ZBCCL_LOG_ERROR("alignment should be power of 2, but real " << alignment);
        return nullptr;
    }

    uint64_t head_skip = 0;
    auto aligned_size = allocated_size_align_up(size);
    MemoryRange anchor{0, aligned_size};

    pthread_spin_lock(&spinlock_);
    auto size_pos = size_idle_tree_.lower_bound(anchor);
    while (size_pos != size_idle_tree_.end() &&
                 !alignment_matches(*size_pos, alignment, aligned_size, head_skip)) {
        ++size_pos;
    }

    if (size_pos == size_idle_tree_.end()) {
        pthread_spin_unlock(&spinlock_);
        ZBCCL_LOG_WARN(
                "cannot allocate with size: " << size << ", alignment: " << alignment);
        return nullptr;
    }

    auto target_offset = size_pos->offset_;
    auto target_size = size_pos->size_;
    auto addr_pos = address_idle_tree_.find(target_offset);
    if (addr_pos == address_idle_tree_.end()) {
        pthread_spin_unlock(&spinlock_);
        ZBCCL_LOG_ERROR("offset(" << target_offset << ") size(" << target_size <<
                                        ") in size tree, not in address tree.");
        return nullptr;
    }
    MemoryRange result_range{size_pos->offset_ + head_skip, aligned_size};
    size_idle_tree_.erase(size_pos);
    address_idle_tree_.erase(addr_pos);

    if (head_skip > 0) {
        size_idle_tree_.emplace(MemoryRange{target_offset, head_skip});
        address_idle_tree_.emplace(target_offset, head_skip);
    }

    if (head_skip + aligned_size < target_size) {
        MemoryRange leftMR{target_offset + head_skip + aligned_size,
                                                target_size - head_skip - aligned_size};
        size_idle_tree_.emplace(leftMR);
        address_idle_tree_.emplace(leftMR.offset_, leftMR.size_);
    }

    address_used_tree_.emplace(result_range.offset_, result_range.size_);
    used_size_ += result_range.size_;
    pthread_spin_unlock(&spinlock_);

    return base_ + result_range.offset_;
}

bool MemoryHeap::changeSize(void *address, uint64_t size) noexcept {
    auto u8a = reinterpret_cast<uint8_t *>(address);
    if (u8a < base_ || u8a >= base_ + size_) {
        ZBCCL_LOG_ERROR("release invalid address " << address);
        return false;
    }

    if (size == 0) {
        release(address);
        return true;
    }

    auto offset = u8a - base_;
    pthread_spin_lock(&spinlock_);
    auto pos = address_used_tree_.find(offset);
    if (pos == address_used_tree_.end()) {
        pthread_spin_unlock(&spinlock_);
        ZBCCL_LOG_ERROR("change size for address " << address << " not allocated.");
        return false;
    }

    // size不变
    if (pos->second == size) {
        pthread_spin_unlock(&spinlock_);
        return true;
    }

    // 缩小size
    if (pos->second > size) {
        reduce_size_in_lock(pos, size);
        used_size_ -= pos->second - size;
        pthread_spin_unlock(&spinlock_);
        return true;
    }

    // 扩大size
    auto success = expend_size_in_lock(pos, size);
    used_size_ += size - pos->second;
    pthread_spin_unlock(&spinlock_);

    return success;
}

int32_t MemoryHeap::release(void *address) noexcept {
    auto u8a = reinterpret_cast<uint8_t *>(address);
    if (u8a < base_ || u8a >= base_ + size_) {
        ZBCCL_LOG_ERROR("release invalid address " << address);
        return Z_ERROR_ALLOC;
    }

    auto offset = u8a - base_;
    pthread_spin_lock(&spinlock_);
    auto pos = address_used_tree_.find(offset);
    if (pos == address_used_tree_.end()) {
        pthread_spin_unlock(&spinlock_);
        ZBCCL_LOG_ERROR("release address " << address << " not allocated.");
        return Z_ERROR_ALLOC;
    }

    auto size = pos->second;
    uint64_t final_offset = static_cast<uint64_t>(offset);
    uint64_t final_size = size;
    address_used_tree_.erase(pos);

    auto prev_addr_pos = address_idle_tree_.lower_bound(offset);
    if (prev_addr_pos != address_idle_tree_.begin()) {
        --prev_addr_pos;
        if (prev_addr_pos != address_idle_tree_.end() &&
                prev_addr_pos->first + prev_addr_pos->second ==
                        static_cast<uint64_t>(offset)) {
            // 合并前一个range
            final_offset = prev_addr_pos->first;
            final_size += prev_addr_pos->second;

            auto prev_addr_range = *prev_addr_pos;
            address_idle_tree_.erase(prev_addr_pos);
            size_idle_tree_.erase(
                    MemoryRange{prev_addr_range.first, prev_addr_range.second});
        }
    }

    auto next_addr_pos = address_idle_tree_.find(offset + size);
    if (next_addr_pos != address_idle_tree_.end()) {    // 合并后一个range
        uint64_t next_addr = next_addr_pos->first;
        uint64_t next_size = next_addr_pos->second;
        final_size += next_size;
        address_idle_tree_.erase(next_addr_pos);
        size_idle_tree_.erase(MemoryRange{next_addr, next_size});
    }
    address_idle_tree_.emplace(final_offset, final_size);
    size_idle_tree_.emplace(MemoryRange{final_offset, final_size});
    used_size_ -= final_size;
    pthread_spin_unlock(&spinlock_);

    return Z_OK;
}

size_t MemoryHeap::getTotalSize() noexcept {
    return size_;
}

size_t MemoryHeap::getInUsedSize() noexcept {
    return used_size_;
}

bool MemoryHeap::allocatedSize(void *address, uint64_t &size) const noexcept {
    auto u8a = reinterpret_cast<uint8_t *>(address);
    if (u8a < base_ || u8a >= base_ + size_) {
        ZBCCL_LOG_ERROR("release invalid address " << address);
        return false;
    }

    auto offset = u8a - base_;
    bool exist = false;
    pthread_spin_lock(&spinlock_);
    auto pos = address_used_tree_.find(offset);
    if (pos != address_used_tree_.end()) {
        exist = true;
        size = pos->second;
    }
    pthread_spin_unlock(&spinlock_);

    return exist;
}

uint64_t MemoryHeap::allocated_size_align_up(uint64_t input_size) noexcept {
    constexpr uint64_t align_size = 16UL;
    constexpr uint64_t align_size_mask = ~(align_size - 1UL);
    return (input_size + align_size - 1UL) & align_size_mask;
}

bool MemoryHeap::alignment_matches(const MemoryRange &mr, uint64_t alignment,
                                                                    uint64_t size,
                                                                    uint64_t &head_skip) noexcept {
    if (mr.size_ < size) {
        return false;
    }

    if ((mr.offset_ & (alignment - 1UL)) == 0UL) {
        head_skip = 0;
        return true;
    }

    auto aligned_offset = ((mr.offset_ + alignment - 1UL) & (~(alignment - 1UL)));
    head_skip = aligned_offset - mr.offset_;
    return mr.size_ >= size + head_skip;
}

void MemoryHeap::reduce_size_in_lock(
        const std::map<uint64_t, uint64_t>::iterator &pos,
        uint64_t new_size) noexcept {
    auto offset = pos->first;
    auto old_size = pos->second;
    pos->second = new_size;
    auto next_addr_pos = address_idle_tree_.find(offset + old_size);
    if (next_addr_pos == address_idle_tree_.end()) {
        address_idle_tree_.emplace(offset + new_size, old_size - new_size);
        size_idle_tree_.emplace(
                MemoryRange{offset + new_size, old_size - new_size});
    } else {
        auto next_size_pos = size_idle_tree_.find(
                MemoryRange{next_addr_pos->first, next_addr_pos->second});
        size_idle_tree_.erase(next_size_pos);
        next_addr_pos->second += (old_size - new_size);
        size_idle_tree_.emplace(
                MemoryRange{next_addr_pos->first, next_addr_pos->second});
    }
}

bool MemoryHeap::expend_size_in_lock(
        const std::map<uint64_t, uint64_t>::iterator &pos,
        uint64_t new_size) noexcept {
    auto offset = pos->first;
    auto old_size = pos->second;
    auto delta = new_size - old_size;

    auto next_addr_pos = address_idle_tree_.find(offset + old_size);
    if (next_addr_pos == address_idle_tree_.end() ||
            next_addr_pos->second < delta) {
        return false;
    }

    pos->second = new_size;
    auto next_size_pos = size_idle_tree_.find(
            MemoryRange{next_addr_pos->first, next_addr_pos->second});
    if (next_addr_pos->second == delta) {
        size_idle_tree_.erase(next_size_pos);
        address_idle_tree_.erase(next_addr_pos);
    } else {
        size_idle_tree_.erase(next_size_pos);
        next_addr_pos->second -= delta;
        size_idle_tree_.emplace(
                MemoryRange{next_addr_pos->first, next_addr_pos->second});
    }

    return true;
}

// DualMemoryHeap
DualMemoryHeap::DualMemoryHeap(void *base, uint64_t size) :
    CustomMemoryHeap(base, size), size_small_(SMALL_SIZE), size_large_(size - SMALL_SIZE),
    small_(base_, size_small_), large_(base_ + SMALL_SIZE, size_large_) {
    if (size < (2 * SMALL_SIZE)) {
        throw std::runtime_error("DualMemoryHeap: size is too small (smaller than 2x default small heap size)");
    }
    initialized_ = true;
}

void *DualMemoryHeap::alignedAllocate(uint64_t alignment, uint64_t size) noexcept {
    if (size >= SMALL_ALLOC) {
        return large_.alignedAllocate(alignment, size);
    }
    else {
        auto ptr = small_.alignedAllocate(alignment, size);
        if (ptr == nullptr && ENABLE_CROSS) {
            ZBCCL_LOG_WARN("[heap] using large pool for small overflow");
            return large_.alignedAllocate(alignment, size);
        }
        return ptr;
    }
}

size_t DualMemoryHeap::getTotalSize() noexcept {
    return size_small_ + size_large_;
}

size_t DualMemoryHeap::getInUsedSize() noexcept {
    return small_.getInUsedSize() + large_.getInUsedSize();
}

int32_t DualMemoryHeap::release(void *address) noexcept {
    auto offset = reinterpret_cast<uint8_t*>(address) - base_;
    ZResult res = Z_ERROR;
    if (offset >= size_small_) {
        res = large_.release(address);
    } else {
        res = small_.release(address);
    }
    return res;
}

bool DualMemoryHeap::allocatedSize(void *address, uint64_t &size) const noexcept {
    auto offset = reinterpret_cast<uint8_t*>(address) - base_;
    if (offset >= size_small_) {
        return large_.allocatedSize(address, size);
    } else {
        return small_.allocatedSize(address, size);
    }
}

}  // namespace heap
}  // namespace sma
}  // namespace zbccl

namespace zbccl {
namespace sma {

ZBCCL_API int HeapAlignedAllocate(void **devPtr, size_t size,
                                  std::shared_ptr<heap::MemoryHeap> shmem_pool) {
    *devPtr = shmem_pool->alignedAllocate(ALIGN_32, size);
    if (*devPtr == nullptr) {
        return Z_ERROR_ALLOC;
    } else {
        return Z_OK;
    }
}

ZBCCL_API int HeapRelease(void *devPtr, std::shared_ptr<heap::MemoryHeap> shmem_pool) {
    return shmem_pool->release(devPtr);
}

ZBCCL_API int GetTotalSize(size_t &size, std::shared_ptr <heap::MemoryHeap> shmem_pool) {
    size = shmem_pool->getTotalSize();
    return Z_OK;
}

ZBCCL_API int GetInUsedSize(size_t &size, std::shared_ptr <heap::MemoryHeap> shmem_pool) {
    size = shmem_pool->getInUsedSize();
    return Z_OK;
}

ZBCCL_API int CustomHeapAlignedAllocate(void **devPtr, size_t size,
                                        std::shared_ptr<heap::CustomMemoryHeap> shmem_pool) {
    if (!shmem_pool) {
        return Z_ERROR;
    }
    *devPtr = shmem_pool->alignedAllocate(ALIGN_32, size);
    if (*devPtr == nullptr) {
        return Z_ERROR_ALLOC;
    } else {
        return Z_OK;
    }
}

ZBCCL_API int CustomHeapRelease(void *devPtr, std::shared_ptr<heap::CustomMemoryHeap> shmem_pool) {
    if (!shmem_pool) {
        return Z_ERROR;
    }
    return shmem_pool->release(devPtr);
}

ZBCCL_API int CustomGetTotalSize(size_t &size, std::shared_ptr <heap::CustomMemoryHeap> shmem_pool) {
    if (!shmem_pool) {
        return Z_ERROR;
    }
    size = shmem_pool->getTotalSize();
    return Z_OK;
}

ZBCCL_API int CustomGetInUsedSize(size_t &size, std::shared_ptr <heap::CustomMemoryHeap> shmem_pool) {
    if (!shmem_pool) {
        return Z_ERROR;
    }
    size = shmem_pool->getInUsedSize();
    return Z_OK;
}

}  // namespace sma
}  // namespace zbccl