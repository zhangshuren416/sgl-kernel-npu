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
#include "zbccl_sma_device.h"

namespace zbccl {
namespace sma {

std::string format_size(uint64_t size)
{
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (size <= 1024) {
        os << size << " bytes";
    } else if (size <= 1048576) {
        os << (size / 1024.0);
        os << " KiB";
    } else if (size <= 1073741824ULL) {
        os << (size / 1048576.0);
        os << " MiB";
    } else {
        os << (size / 1073741824.0);
        os << " GiB";
    }
    return os.str();
}

ZEvent EventPool::get(int device)
{
    ZBCCL_ASSERT_S(0 <= device, "get device error:", Z_INVALID_VALUE);
    ZBCCL_ASSERT_S(device < static_cast<int>(pools_.size()), "get device error:", Z_INVALID_VALUE);
    auto &pool = pools_[device];
    auto destructor = [&pool](c10_npu::NPUEvent *event) {
        std::lock_guard<std::mutex> g(pool.mutex_);
        pool.event_pool_.push_back(std::unique_ptr<c10_npu::NPUEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
        std::lock_guard<std::mutex> g(pool.mutex_);
        if (!pool.event_pool_.empty()) {
            auto *event = pool.event_pool_.back().release();
            pool.event_pool_.pop_back();
            return ZEvent(event, destructor);
        }
    }
    // otherwise, allocate a new event that will be returned to the pool on destruction.
    return ZEvent(std::make_unique<c10_npu::NPUEvent>(ACL_EVENT_CAPTURE_STREAM_PROGRESS).release(), destructor);
}

void EventPool::emptyCache()
{
    for (auto &pool : pools_) {
        std::lock_guard<std::mutex> g(pool.mutex_);
        pool.event_pool_.clear();
    }
}

}  // namespace sma
}  // namespace zbccl


namespace zbccl {
namespace sma {
namespace device {

using Blocks = std::set<DeviceBlock *, Comparison>;
using StreamSet = ska::flat_hash_set<c10_npu::NPUStream>;

// private funcs
DeviceBlock *DeviceSMACachingAllocator::alloc_found_block(DeviceAllocParams params, size_t orig_size, std::shared_ptr<c10::GatheredContext> context,
                                                          bool split_remainder, uint8_t allocator_type) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool_;
    auto stream = params.stream();

    ZBCCL_ASSERT_S(params.result_ == Z_OK && params.block_ != nullptr && params.block_->ptr_ != nullptr, Z_INVALID_PTR);
    DeviceBlock *block = params.block_;
    DeviceBlock *remaining = nullptr;

    // const bool already_split = block->isSplit();
    if (split_remainder) {
        remaining = block;

        block = new DeviceBlock(device, stream, size, pool, block->ptr_, block->block_type_);
        block->prev_ = remaining->prev_;
        if (block->prev_) {
            block->prev_->next_ = block;
        }
        block->next_ = remaining;

        // consider that size & begin & end is already aligned, no need to do re-aligned for new block
        remaining->prev_ = block;
        remaining->ptr_ = static_cast<char *>(remaining->ptr_) + size;
        remaining->size_ -= size;

        pool->insertBlock(remaining->block_type_, remaining);
    }

    block->allocated_ = true;
    block->requested_size_ = orig_size;
    if (!block->is_safe_ ) {
        ZBCCL_LOG_WARN("Unsafe memory block is passively refreshed by releasing and allocating memory again");
    }
    block->is_safe_ = true;

    block->context_when_allocated_ = std::move(context);

    active_blocks_.insert(block);

    ZBCCL_LOG_DEBUG("PTA CachingAllocator malloc: malloc = " << block->size_);

    return block;
}

std::vector<const DeviceBlock *> DeviceSMACachingAllocator::get_all_blocks() const {
    std::vector<const DeviceBlock *> blocks;
    blocks.insert(blocks.end(), default_pool_.small_blocks_.begin(), default_pool_.small_blocks_.end());
    blocks.insert(blocks.end(), default_pool_.large_blocks_.begin(), default_pool_.large_blocks_.end());
    //for (const auto &gp : graph_pools_) {
    //    blocks.insert(blocks.end(), gp.second->small_blocks_.begin(), gp.second->small_blocks_.blocks.end());
    //    blocks.insert(blocks.end(), gp.second->large_blocks_.begin(), gp.second->large_blocks_.blocks.end());
    //}
    blocks.insert(blocks.end(), active_blocks_.begin(), active_blocks_.end());
    return blocks;
}

void DeviceSMACachingAllocator::free_block(DeviceBlock *block, const std::shared_ptr<c10::GatheredContext> &context, uint8_t allocator_type) {
    ZBCCL_ASSERT_S(!block->allocated_ && block->event_count_ == 0, Z_INVALID_VALUE);

    block->context_when_allocated_ = nullptr;
    //size_t original_block_size = block->size_;
    //auto orig_block_ptr = block->ptr_;
    //size_t requested_size = block->requested_size_;

    auto &pool = *block->pool_;

    const std::array<DeviceBlock *, 2> merge_candidates = { block->prev_, block->next_ };
    for (DeviceBlock *merge_candidate : merge_candidates) {
        const int64_t subsumed_size = static_cast<int64_t>(try_merge_blocks(block, merge_candidate, pool));
    }

    active_blocks_.erase(block);
    pool.insertBlock(block->block_type_, block);
}

size_t DeviceSMACachingAllocator::try_merge_blocks(DeviceBlock *dst, DeviceBlock *src, DeviceBlockPool &pool) {
    if (!src || src->allocated_ || src->event_count_ > 0 || !src->stream_uses_.empty()) {
        return 0;
    }

    ZBCCL_ASSERT_S(dst->isSplit() && src->isSplit(), "assert is_split error:", Z_INVALID_VALUE);

    if (dst->prev_ == src) {
        dst->ptr_ = src->ptr_;
        dst->prev_ = src->prev_;
        if (dst->prev_) {
            dst->prev_->next_ = dst;
        }
    } else {
        dst->next_ = src->next_;
        if (dst->next_) {
            dst->next_->prev_ = dst;
        }
    }

    const size_t subsumed_size = src->size_;
    dst->size_ += subsumed_size;

    ZBCCL_ASSERT_S(!pool.is_private_, "assert pool !is_private error:", Z_INVALID_VALUE);
    pool.eraseBlock(src->block_type_, src);

    delete src;
    src = nullptr;

    return subsumed_size;
}

DeviceBlockPool &DeviceSMACachingAllocator::get_pool(size_t size, aclrtStream stream, DeviceBlockType &block_type) {
    // captures_underway is a conservative guess that the current stream may be
    // capturing. It's only non-empty if some thread has begun and not yet ended
    // a capture, so it's usually 0, and we can short-circuit
    // npuStreamCaptureStatus (which does a TLS lookup).
    /*
    if (C10_UNLIKELY(!captures_underway_.empty())) {
        for (auto &entry : captures_underway) {
            if (entry.second(stream)) {
                auto it1 = graph_pools.find(entry.first);
                TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
                if (size <= kSmallSize) {
                    return it1->second->small_blocks;
                } else {
                    return it1->second->large_blocks;
                }
            }
        }
    }*/
    if (size <= kSmallSize) {
        block_type = BT_SMALL;
    } else {
        block_type = BT_BIG;
    }
    return default_pool_;
}

bool DeviceSMACachingAllocator::should_split(const DeviceBlock *block, size_t size) {
    size_t remaining = block->size_ - size;
    if (block->block_type_ == BT_SMALL) {
        return remaining >= kMinBlockSize;
    } else {
        return (size < SMAConfig::max_split_size()) && (remaining > kSmallSize);
    }
}

bool DeviceSMACachingAllocator::get_free_block(DeviceAllocParams &p) {
    DeviceBlockPool &pool = *p.pool_;

    Blocks block_slot = (p.block_type_ == BT_SMALL) ? pool.small_blocks_ : pool.large_blocks_;
    if (ZBCCL_UNLIKELY(set_fraction_ && SMAConfig::garbage_collection_threshold() > 0.0)) {
        // Track block reuse interval only when garbage collection is enabled.
        for (auto &b : block_slot) {
            ++b->gc_count_;
        }
    }
    auto it = block_slot.lower_bound(&p.search_key_);
    // stream diff indicate that block in this stream have already no appropriate block
    if (it == block_slot.end() || (*it)->stream_ != p.stream()) {
        return false;
    }

    // Do not return an over-sized block for a large request
    if ((p.size() < SMAConfig::max_split_size()) &&
        ((*it)->size_ >= SMAConfig::max_split_size())) {
        return false;
    }
    // Allow over-sized block size to be rounded up but within a limit
    if ((p.size() >= SMAConfig::max_split_size()) && ((*it)->size_ >= p.size() + kLargeBuffer)) {
        return false;
    }
    p.block_ = *it;
    p.block_type_ = (*it)->block_type_;
    (*it)->gc_count_ = 0; // Denote this block has been used
    pool.eraseBlock(p.block_type_, *it);
    return true;
}

void DeviceSMACachingAllocator::garbage_collect_cached_blocks(const std::shared_ptr<c10::GatheredContext>& ctx,
                                                              std::unique_lock<std::recursive_mutex>& lock) {
    // Free unused cached blocks to reclaim NPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold =
            static_cast<size_t>(SMAConfig::garbage_collection_threshold() * allowed_memory_maximum_);
    // No need to trigger GC yet
    if (total_allocated_memory_ <= gc_threshold) {
        return;
    }
    const auto target_size = total_allocated_memory_ - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto &b : default_pool_.large_blocks_) {
        if (!b->isSplit()) {
            total_age += b->gc_count_;
            ++freeable_block_count;
        }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
        return;
    }

    {
        UnlockGuard guard(lock);
        c10_npu::npuSynchronizeDevice(true);
    }

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed && freeable_block_count > 0) {
        // Free blocks exceeding this age threshold first.
        double age_threshold = total_age / freeable_block_count;
        // Stop iteration if we can no longer free a block.
        block_freed = false;

        // Free blocks of > avg age. Don't stop upon reaching the target_size,
        // we don't want this GC to be triggered frequently.
        auto it = default_pool_.large_blocks_.begin();
        while (it != default_pool_.large_blocks_.end()) {
            DeviceBlock *block = *it;
            ++it;
            if (!block->isSplit() && block->gc_count_ >= age_threshold) {
                block_freed = true;
                gc_reclaimed += block->size_;
                total_age -= block->gc_count_; // Decrement the age
                freeable_block_count--;       // One less block that can be freed
                release_block(block, ctx);

                ZBCCL_LOG_DEBUG("SMACachingAllocator gc: free = " << block->size_ <<
                                                                  " allocated = " << total_allocated_memory_);
            }
        }
    }
}

bool DeviceSMACachingAllocator::alloc_block(DeviceAllocParams &p, bool isRetry, const std::shared_ptr<c10::GatheredContext> &ctx,
                                            std::unique_lock<std::recursive_mutex> &lock) {
    size_t size = p.alloc_size_;
    void *ptr = nullptr;

    if (set_fraction_ && total_allocated_memory_ + size > allowed_memory_maximum_) {
        p.result_ = Z_ERROR_ALLOC;
        return false;
    } else {
        // TODO add active_pool is_private check before this
        if (mem_heap_inited_) {
            p.result_ = zbccl::sma::HeapAlignedAllocate(&ptr, size, mem_heap_pool_);
            if (p.result_ == Z_OK) {
                shmem_addrs_.insert(ptr);
            }
        } else {
            ZBCCL_LOG_ERROR("sma heap not inited, using aclRT instead(this may be a undefined behavior)");
            // p.result_ = aclrtMallocAlign32(&ptr, size, policy);
        }

        if (p.result_ != Z_OK) {
            return false;
        }
    }

    /*
    if (p.pool_->is_private_) {
        // The block is for a NPU graph's PrivatePool.
        p.pool_->npuMalloc_count++;
    }*/

    total_allocated_memory_ += size;
    p.block_ = new DeviceBlock(p.device(), p.stream(), size, p.pool_, (char *)ptr, p.block_type_);
    ZBCCL_LOG_DEBUG("DeviceSMACachingAllocator: malloc = " << size << " ret = " << p.result_);

    // p.block_ came from new, not npuMalloc. It should not be nullptr here.
    ZBCCL_ASSERT_S(p.block_ != nullptr && p.block_->ptr_ != nullptr, "block invalid!");

    p.block_->context_when_allocated_ = ctx;
    return true;
}

bool DeviceSMACachingAllocator::release_available_cached_blocks(const DeviceAllocParams& p, const std::shared_ptr<c10::GatheredContext>& ctx,
                                                                std::unique_lock<std::recursive_mutex>& lock) {
    // meaning no split over max_split_size, just skip
    if (SMAConfig::max_split_size() == std::numeric_limits<size_t>::max()) {
        return false;
    }
    DeviceBlockPool &pool = *p.pool_;
    DeviceBlock key = p.search_key_;
    key.size_ = (key.size_ < SMAConfig::max_split_size()) ? SMAConfig::max_split_size() : key.size_;
    // FIXME maybe this max_split_size is only appropriate for small pool?
    auto& block_slot = (p.block_type_ == BT_SMALL) ? pool.small_blocks_ : pool.large_blocks_;
    auto it = block_slot.lower_bound(&key);

    {
        UnlockGuard guard(lock);
        c10_npu::npuSynchronizeDevice(true);
    }

    if (it == block_slot.end() || (*it)->stream_ != p.stream()) {
        // No single block is large enough; free multiple oversize blocks, starting with the largest
        if (it == block_slot.begin()) {
            return false;
        }
        size_t totalReleased = 0;
        // Back up one item.  Now on the largest block for the correct stream
        --it;
        while ((totalReleased < key.size_) && ((*it)->size_ >= SMAConfig::max_split_size()) &&
               ((*it)->stream_ == p.stream())) {
            auto cur = it;
            totalReleased += (*it)->size_;
            if (it != block_slot.begin()) {
                --it;
                release_block(*cur, ctx);
                if (totalReleased >= key.size_) break;
            } else {
                release_block(*cur, ctx);
                break;
            }
        }
        if (totalReleased < key.size_) {
            return false;
        }
    } else {
        // free a single block large enough and return
        release_block(*it, ctx);
    }
    return true;
}

void DeviceSMACachingAllocator::release_block(DeviceBlock *block, const std::shared_ptr<c10::GatheredContext> &context) {
    if (shmem_addrs_.count((void *)block->ptr_)) {
        ZBCCL_CHECK_S(zbccl::sma::HeapRelease((void *)block->ptr_, mem_heap_pool_) == ACL_SUCCESS, "shmem heap free failed");
    } else {
        ZBCCL_LOG_ERROR("sma miss this ptr, using aclRT instead(this may be a undefined behavior)");
        // aclrtFree((void *)block->ptr_);
    }
    total_allocated_memory_ -= block->size_;

    auto *pool = block->pool_;
    /*
    if (pool->is_private_) {
        // The npuFreed block belonged to a NPU graph's PrivatePool.
        ZBCCL_ASSERT_S(pool->npuMalloc_count_ > 0);
        pool->npuMalloc_count_--;
    }*/
    ZBCCL_LOG_DEBUG("DeviceSMACachingAllocator free by: size= " << block->size_);

    pool->eraseBlock(block->block_type_, block);
    delete block;
    block = nullptr;
}

void DeviceSMACachingAllocator::release_pool(DeviceBlockPool &pool, const std::shared_ptr<c10::GatheredContext> &context, bool free_private)
{
    // Frees all non-split blocks
    // skip private pool if not free_private
    if (!free_private && pool.is_private_) return;

    auto it = pool.large_blocks_.begin();
    while (it != pool.large_blocks_.end()) {
        DeviceBlock *block = *it;
        ++it;
        if (!block->prev_ && !block->next_) {
            release_block(block, context);
        }
    }

    it = pool.small_blocks_.begin();
    while (it != pool.small_blocks_.end()) {
        DeviceBlock *block = *it;
        ++it;
        if (!block->prev_ && !block->next_) {
            release_block(block, context);
        }
    }
}

ZEvent DeviceSMACachingAllocator::create_event_internal(int idx)
{
    // Leak the event pool to avoid shutdown issues.
    static auto *event_pool_ = new EventPool();
    return event_pool_->get(idx);
}

void DeviceSMACachingAllocator::synchronize_and_free_events(bool check_error, const std::shared_ptr<c10::GatheredContext> &context)
{
    // This function syncs, so capture should not be underway. Might as well
    // make sure capture-deferred end of life events get processed too.
    //ZBCCL_ASSERT_S(captures_underway_.empty());
    //insert_events_deferred_until_no_capture(context);

    // Synchronize on outstanding events and then free associated blocks.
    for (auto &st : npu_events_) {
        for (auto &e : st.second) {
            ZEvent event = std::move(e.first);
            DeviceBlock *block = e.second;
            auto err = aclrtSynchronizeEvent(*event);
            if (err != ACL_SUCCESS) {
                ZBCCL_LOG_ERROR("Event: aclrtSynchronizeEvent failed, event = " << event.get());
            } else {
                ZBCCL_LOG_INFO("Event: aclrtSynchronizeEvent is successfully executed, event = " << event.get());
            }

            block->event_count_--;
            if (block->event_count_ == 0) {
                free_block(block, context);
            }
        }
    }
    npu_events_.clear();
}

bool DeviceSMACachingAllocator::release_cached_blocks(bool check_error, const std::shared_ptr<c10::GatheredContext> &context) {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(check_error, context);

    // Free all non-split cached blocks, including graph pools which use_count is down to 0
    release_pool(default_pool_, context, false);
    /*
    for (auto it = graph_pools_freeable.begin(); it != graph_pools_freeable.end();) {
        // See notifyCaptureDestroy for the strategy here.
        TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
        release_blocks(it->second->small_blocks, context, free_physical);
        release_blocks(it->second->large_blocks, context, free_physical);
        if (it->second->npuMalloc_count == 0) {
            auto erase_count = graph_pools.erase(it->first);
            TORCH_INTERNAL_ASSERT(erase_count == 1);
            it = graph_pools_freeable.erase(it);
        } else {
            ++it;
        }
    }*/

    return true;
}

void DeviceSMACachingAllocator::insert_events(DeviceBlock *block)
{
    int pre_device = -1;
    c10_npu::GetDevice(&pre_device);
    aclrtContext compiler_ctx = aclrtContext();
    aclError ret_ctx = aclrtGetCurrentContext(&compiler_ctx);

    StreamSet streams(std::move(block->stream_uses_));
    ZBCCL_ASSERT_S(block->stream_uses_.empty(), "check remain stream is empty failed:", Z_INVALID_VALUE);
    for (auto &stream : streams) {
        ZBCCL_CHECK_S(c10_npu::SetDevice(stream.device_index()) == ACL_SUCCESS, "c10_npu func failed");

        ZEvent event = create_event_internal(stream.device_index());
        event->record(stream);
        ZBCCL_LOG_INFO("Event: record DeviceAllocator is successfully executed, event = " << event.get());

        block->event_count_++;
        npu_events_[stream].emplace_back(std::move(event), block);
    }
    if (ret_ctx == ACL_SUCCESS) {
        ZBCCL_CHECK_S(aclrtSetCurrentContext(compiler_ctx) == ACL_SUCCESS, "c10_npu func failed");
        // Setting context will exchange device implicitly, so we need to reset the cached device here to ensure consistency.
        ZBCCL_CHECK_S(c10_npu::SetDevice(pre_device) == ACL_SUCCESS, "c10_npu func failed");
    }
}

void DeviceSMACachingAllocator::process_events(const std::shared_ptr<c10::GatheredContext> &context) {
    //insert_events_deferred_until_no_capture(context);

    // Process outstanding npuEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    for (auto it = npu_events_.begin(); it != npu_events_.end();) {
        while (!it->second.empty()) {
            auto &e = it->second.front();
            ZEvent event = std::move(e.first);
            DeviceBlock *block = e.second;

            if (!event->query()) {
                e.first = std::move(event);
                break;
            }

            block->event_count_--;
            if (block->event_count_ == 0) {
                free_block(block, context);
            }
            it->second.pop_front();
        }

        if (it->second.empty()) {
            it = npu_events_.erase(it);
        } else {
            it++;
        }
    }
}

void DeviceSMACachingAllocator::cache_info_aux(DeviceBlockPool &block_pool, size_t *total, size_t *largest) {
    for (auto it = block_pool.small_blocks_.begin(); it != block_pool.small_blocks_.end(); ++it) {
        size_t blocksize = (*it)->size_;
        *total += blocksize;
        if (blocksize > *largest) {
            *largest = blocksize;
        }
    }
    for (auto it = block_pool.large_blocks_.begin(); it != block_pool.large_blocks_.end(); ++it) {
        size_t blocksize = (*it)->size_;
        *total += blocksize;
        if (blocksize > *largest) {
            *largest = blocksize;
        }
    }
}

size_t DeviceSMACachingAllocator::round_size(size_t size) {
    size = size + 32;
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
}

size_t DeviceSMACachingAllocator::get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
        return kSmallBuffer;
    } else if (size <= kLargeBuffer) {
        return kLargeBuffer;
    } else {
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
}

// public funcs
void DeviceSMACachingAllocator::releaseAndFreeEvents() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    std::shared_ptr<c10::GatheredContext> context = nullptr;
    for (auto &st : npu_events_) {
        for (auto &e : st.second) {
            ZEvent event_ = std::move(e.first);
            DeviceBlock *block = e.second;
            block->event_count_--;
            if (block->event_count_ == 0) {
                free_block(block, context);
            }
        }
    }
    npu_events_.clear();
}

void DeviceSMACachingAllocator::markAllBlockUnsafe() {
    for (auto &active_block : active_blocks_) {
        active_block->is_safe_ = false;
    }
    return;
}

void *DeviceSMACachingAllocator::getBaseAllocation(DeviceBlock *block, size_t *outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    while (block->prev_) {
        block = block->prev_;
    }
    void *basePtr = block->ptr_;
    if (outSize) {
        size_t size = 0;
        while (block) {
            size += block->size_;
            block = block->next_;
        }
        *outSize = size;
    }
    return basePtr;
}



DeviceBlock *DeviceSMACachingAllocator::malloc(int device, size_t orig_size, aclrtStream stream, uint8_t allocator_type) {
    // done outside the lock because we don't know what locks the recorder needs to have...
    //auto context = maybeGatherContext(RecordContext::STATE);
    auto context = nullptr;

    std::unique_lock <std::recursive_mutex> lock(mutex_);

    if (device == -1) {
        ZBCCL_CHECK_S(c10_npu::GetDevice(&device) == ACL_SUCCESS, "c10_npu func check failed!");
    }

    process_events(context);
    /*
    if (C10_LIKELY(captures_underway_.empty())) {
        // Processes end-of-life events for outstanding allocations used on
        // multiple streams (checks if their NPU-side uses are complete and
        // recycles their memory if so)
        //
        // Q. Why skip process_events if a capture might be underway?
        // A. process_events involves npuEventQueries, illegal during NPU graph
        //    capture.
        //    Dumb simple solution: defer reclaiming these allocations until after
        //    capture. Cross-stream memory use is uncommon, so the deferral's
        //    effect on memory use during capture should be small.
        process_events(context);
    }*/
    auto size = round_size(orig_size);
    DeviceBlockType block_type;
    auto &pool = get_pool(size, stream, block_type);
    const size_t alloc_size = get_allocation_size(size);

    DeviceAllocParams params(device, size, stream, &pool, alloc_size, block_type);

    // First, try to get a block from the existing pool.
    bool block_found = get_free_block(params);
    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
        // Do garbage collection if the flag is set.
        if (ZBCCL_UNLIKELY(set_fraction_ && SMAConfig::garbage_collection_threshold() > 0.0)) {
            garbage_collect_cached_blocks(context, lock);
        }
        // Attempt allocate
        block_found = alloc_block(params, false, context, lock) ||
                      // Free enough available cached blocks to satisfy alloc and retry alloc.
                      (release_available_cached_blocks(params, context, lock) &&
                       alloc_block(params, false, context, lock));
    }

    // if (!block_found && ZBCCL_UNLIKELY(captures_underway_.empty())) {
    if (!block_found) {
        ZBCCL_LOG_WARN(
                "Get a block from the existing pool failed. Try to free cached blocks and reallocate. This warning log can be ignored.");
        // Free all non-split cached blocks and retry alloc.
        {
            UnlockGuard guard(lock);
            // Make sure taskqueue is empty, then execute release_cached_blocks
            c10_npu::npuSynchronizeDevice(true);
        }
        // TODO fix context & free_phy bool
        block_found = (release_cached_blocks(true, nullptr) && alloc_block(params, true, context, lock));
    }

    if (!block_found) {
        if (params.result_ == Z_NOT_ENOUGH_MEM) {
            // TODO fulfill current OOM state to be comparable to original allocator
            ZBCCL_LOG_ERROR("NPU out of memory. Tried to allocate " << format_size(alloc_size) << " (NPU:" << device
                                                                    << "); with " << format_size(total_allocated_memory_)
                                                                    << " total allocated. ");
        }
        ZBCCL_CHECK_S(params.result_ == Z_OK, "check alloc result failed");
    }

    bool split_remainder = should_split(params.block_, params.size());
    return alloc_found_block(std::move(params), orig_size, std::move(context), split_remainder, allocator_type);
}

void DeviceSMACachingAllocator::free(DeviceBlock *block, uint8_t allocator_type) {
    std::shared_ptr<c10::GatheredContext> context = nullptr;
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    block->allocated_ = false;

    // following logic might modify underlying Block, causing the size changed. We store ahead for reporting
    //auto orig_block_ptr = block->ptr_;
    auto orig_block_size = block->size_;
    //auto orig_block_type = block->block_type_;

    if (!block->stream_uses_.empty() && c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        /*
        if (C10_UNLIKELY(!captures_underway.empty())) {
            // It's forbidden to npuEventQuery an event recorded during NPU graph
            // capture. We conservatively defer recording end-of-life events until
            // the next call to process_events() (which won't happen until no
            // captures are underway)
            needs_events_deferred_until_no_capture.push_back(block);
        } else {
            insert_events(block);
        }*/
        insert_events(block);
    } else {
        free_block(block, context, allocator_type);
    }

    ZBCCL_LOG_INFO("PTA CachingAllocator free: free = " << orig_block_size << ", allocated = " << total_allocated_memory_);
}

void DeviceSMACachingAllocator::recordStream(DeviceBlock *block, c10_npu::NPUStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    block->stream_uses_.insert(stream);
    /*
    if (C10_UNLIKELY(!captures_underway.empty())) {
        block_to_npugraph_stream_uses[block].insert(stream);
    }*/
}

// this func is a non-standard func since Pytorch do not have this API, and erase without query is a wrong action
void DeviceSMACachingAllocator::eraseStream(DeviceBlock *block, c10_npu::NPUStream stream) {
    std::shared_ptr<c10::GatheredContext> context = nullptr;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    block->stream_uses_.erase(stream);

    // free block, lazy destroy block related events
    for (auto it = npu_events_[stream].begin(); it != npu_events_[stream].end();) {
        if (block != it->second) {
            it++;
            continue;
        }
        it = npu_events_[stream].erase(it);
        block->event_count_--;
        if (block->event_count_ == 0) {
            free_block(block, context);
            break;
        }
    }
}

void DeviceSMACachingAllocator::setMemoryFraction(double fraction)
{
    size_t device_total;
    zbccl::sma::ReservedTotalSize(device_total, mem_heap_pool_);
    allowed_memory_maximum_ = static_cast<size_t>(fraction * device_total);
    set_fraction_ = true;
}

void DeviceSMACachingAllocator::emptyCache(int device, bool check_error) {
    std::shared_ptr<c10::GatheredContext> context = nullptr;
    // Make sure event deque from taskqueue, then synchronize Event
    c10_npu::npuSynchronizeDevice(check_error);
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    release_cached_blocks(check_error, context);
}

void DeviceSMACachingAllocator::cacheInfo(size_t *total, size_t *largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    cache_info_aux(default_pool_, total, largest);
    /*
    for (const auto &gp : graph_pools) {
        cache_info_aux(gp.second->large_blocks, total, largest);
        cache_info_aux(gp.second->small_blocks, total, largest);
    }*/
}

}  // namespace device
}  // namespace sma
}  // namespace zbccl
