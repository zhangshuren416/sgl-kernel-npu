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
#ifndef ZBCCL_SMA_DEVICE_ALLOC_H
#define ZBCCL_SMA_DEVICE_ALLOC_H

#include "zbccl_sma_common.h"
#include "zbccl_sma_config.h"
#include "zbccl_sma_device_pool.h"

using ZEvent = std::unique_ptr<c10_npu::NPUEvent, std::function<void(c10_npu::NPUEvent *)>>;

namespace zbccl {
namespace sma {

class EventPool {
public:
    // Explicit device count
    EventPool() : pools_(c10_npu::device_count()) {}

    ZEvent get(int device);

    void emptyCache();

private:
    // this struct is too simple to drop in device_pool file
    struct PerDevicePool {
        alignas(64) std::mutex mutex_;
        std::vector<std::unique_ptr<c10_npu::NPUEvent>> event_pool_;
    };
    std::vector<PerDevicePool> pools_;
};

// To prevent the deadlock situation, temporarily release the lock.
//
// Deadlock Scenario Description:
//
// 1. Main Thread:
//    - Acquires the lock and performs sync to clear the taskqueue.
//    - taskqueue wait a empty signal from the sub-thread.
//
// 2. Sub-thread:
//    - Python function (tbe op compile) called in CANN may trigger GC that introduces a resource release operation.
//    - The release operation (`free`) cannot acquire the same lock holded in main thread.
//    - Unable to send a signal to the main thread.
class UnlockGuard {
public:
    explicit UnlockGuard(std::unique_lock<std::recursive_mutex>& lock) : lock_(lock) { lock_.unlock(); }

    ~UnlockGuard() { lock_.lock(); }

private:
    std::unique_lock<std::recursive_mutex>& lock_;
};

}  // namespace sma
}  // namespace zbccl


namespace zbccl {
namespace sma {
namespace device {

class DeviceSMACachingAllocator {
public:
    // TODO: move this(shm-vmm heap class for shmem) into private
    std::shared_ptr<heap::MemoryHeap> mem_heap_pool_{nullptr};
    bool mem_heap_inited_ = false;
    void *shmem_base_addr_ = nullptr;

private:
    // lock around all operations
    mutable std::recursive_mutex mutex_;

    // global unallocated cached blocks(except private_pool)
    DeviceBlockPool default_pool_;

    // allocated or in use by a stream
    ska::flat_hash_set<DeviceBlock *> active_blocks_;

    // TODO: merge into DeviceStats later(addrs allocated by shmem)
    ska::flat_hash_set<void *> shmem_addrs_;

    // outstanding acl events
    ska::flat_hash_map<c10_npu::NPUStream, std::deque<std::pair<ZEvent, DeviceBlock *>>> npu_events_;

    // record used memory.
    size_t total_allocated_memory_ = 0;

    // record maximum allowed memory.
    size_t allowed_memory_maximum_ = 0;

    bool set_fraction_ = false;
    // bool record_history = false;

    // captures_underway tracks if we are diverting some
    // allocations to a specific pool.
    // Most of the time it's empty, in which case malloc can avoid calling
    // aclrtStreamGetCaptureInfo in the hot path.
    //std::vector<std::pair<MempoolId_t, std::function<bool(aclrtStream)>>> captures_underway_;

    // See free() for this thing's purpose
    //std::vector<DeviceBlock *> needs_events_deferred_until_no_capture_;

    // Private pools for NPU graphs
    //ska::flat_hash_map<MempoolId_t, std::unique_ptr<DeviceBlockPool>, MempoolIdHash> graph_pools_;

    // Pools no longer referenced by any graph. Their BlockPools are eligible for
    // free_blocks. Can't be a vector or deque because we might erase entries in
    // any order. Could be an std::list, but we don't care much, access and
    // insert/erase are rare.
    //ska::flat_hash_map<MempoolId_t, DeviceBlockPool*, MempoolIdHash> graph_pools_freeable_;

    // mapping from block to a stream_set, containing streams on which the block
    // was used while npugraph capturing
    //std::unordered_map<DeviceBlock *, stream_set> block_to_npugraph_stream_uses_;

    // All following private methods do not acquire the allocator mutex
    // move a founded block from pool into active_list, may get new block which split from found one
    DeviceBlock *alloc_found_block(DeviceAllocParams params, size_t orig_size, std::shared_ptr<c10::GatheredContext> context,
                                    bool split_remainder, uint8_t allocator_type);

    // get all blocks(in default_pool, graph_pools, and active blocks)
    std::vector<const DeviceBlock *> get_all_blocks() const;

    //std::vector<DeviceBlock*> get_private_pool_head_blocks(DevicePoolPtr pool) const;

    // free a block from active blocks into the pool of cached free blocks
    void free_block(DeviceBlock *block, const std::shared_ptr<c10::GatheredContext> &context, uint8_t allocator_type = 0);

    // combine previously split blocks. returns the size of the subsumed block, or 0 on failure.
    size_t try_merge_blocks(DeviceBlock *dst, DeviceBlock *src, DeviceBlockPool &pool);

    // get running pool & its correspond block_type
    DeviceBlockPool &get_pool(size_t size, aclrtStream stream, DeviceBlockType &block_type);

    // whether split new block by actual size while doing alloc_found
    bool should_split(const DeviceBlock *block, size_t size);

    // search the closest free block in pool from DeviceAllocParams
    bool get_free_block(DeviceAllocParams &p);

    // free unused block until gc_threshold according to gc_count_
    void garbage_collect_cached_blocks(const std::shared_ptr<c10::GatheredContext>& ctx,
                                       std::unique_lock<std::recursive_mutex>& lock);

    // alloc a new block(or map a new one if using expand), need to mark private or not
    bool alloc_block(DeviceAllocParams &p, bool isRetry, const std::shared_ptr<c10::GatheredContext> &ctx,
                     std::unique_lock<std::recursive_mutex> &lock);

    // Free one or more blocks to the system allocator. But only enough to satisfy the target size
    // Start from over max_split_size to small size
    bool release_available_cached_blocks(const DeviceAllocParams& p, const std::shared_ptr<c10::GatheredContext>& ctx,
                                         std::unique_lock<std::recursive_mutex>& lock);

    // npuSynchronizeDevice must be executed before this function can be called
    // TODO remove physical for sma
    bool release_cached_blocks(bool check_error, const std::shared_ptr<c10::GatheredContext> &context, bool free_physical);

    // release a block from cached pool
    void release_block(DeviceBlock *block, const std::shared_ptr<c10::GatheredContext> &context);

    // release all block in pool, also free private pool if free_private
    void release_pool(DeviceBlockPool &pool, const std::shared_ptr<c10::GatheredContext> &context, bool free_private);

    ZEvent create_event_internal(int idx);

    // synchronize on all outstanding events and then free associated blocks.
    void synchronize_and_free_events(bool check_error, const std::shared_ptr<c10::GatheredContext> &context);

    // void remove_npugraph_stream_uses(DeviceBlock *block);

    // insert events for all streams using this activated block
    void insert_events(DeviceBlock *block);

    // void insert_events_deferred_until_no_capture(const std::shared_ptr<c10::GatheredContext> &context);

    // check each stream's events, process only one case if queried ended, decrease block event_count_ and free when it down to 0
    void process_events(const std::shared_ptr<c10::GatheredContext> &context);

    // Accumulates sizes of all memory blocks for given device in given pool
    void cache_info_aux(DeviceBlockPool &block_pool, size_t *total, size_t *largest);

    static size_t round_size(size_t size);
    static size_t get_allocation_size(size_t size);

public:
    DeviceSMACachingAllocator() : default_pool_(false)
    {}

    DeviceBlock *malloc(int device, size_t orig_size, aclrtStream stream, uint8_t allocator_type = 0);

    void free(DeviceBlock *block, uint8_t allocator_type = 0);

    void recordStream(DeviceBlock *block, c10_npu::NPUStream stream);

    void eraseStream(DeviceBlock *block, c10_npu::NPUStream stream);

    // change all activated block into unsafe
    void markAllBlockUnsafe();

    // get head block, along with cum-summed head to tail total segment blocks size
    void *getBaseAllocation(DeviceBlock *block, size_t *outSize);

    // set memory fraction to limit maximum allocated memory
    void setMemoryFraction(double fraction);

    // returns cached blocks to the system allocator
    void emptyCache(int device, bool check_error, bool free_physical);

    // Retrieves info (total size + largest block) of the memory cache
    void cacheInfo(size_t *total, size_t *largest);

    // free all event if count down to 0
    void releaseAndFreeEvents();

    // Called by NPUGraph::capture_begin
    // void beginAllocateToPool(MempoolId_t mempool_id, std::function<bool(aclrtStream)> filter);

    // Called by NPUGraph::capture_end
    //void endAllocateToPool(MempoolId_t mempool_id);

    // Called by NPUGraph::reset
    // void releasePool(MempoolId_t mempool_id);
};

}  // namespace device
}  // namespace sma
}  // namespace zbccl


#endif  // ZBCCL_SMA_DEVICE_ALLOC_H
