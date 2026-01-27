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
#include "zbccl_sma.h"

namespace zbccl {
namespace sma {

SecondaryMemoryAllocator::SecondaryMemoryAllocator() {
}

void SecondaryMemoryAllocator::add_allocated_block(device::DeviceBlock *block) {
    std::lock_guard<std::mutex> lock(mutex_);
    allocated_blocks_[block->ptr_] = block;
}

device::DeviceBlock *SecondaryMemoryAllocator::get_allocated_block(void *ptr, bool remove) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
        return nullptr;
    }
    device::DeviceBlock *block = it->second;
    if (remove) {
        allocated_blocks_.erase(it);
    }
    return block;
}

bool SecondaryMemoryAllocator::initialized() {
    return !device_allocator_.empty();
}

void SecondaryMemoryAllocator::cleanEvent() {
    int count = static_cast<int>(device_allocator_.size());
    for (int i = 0; i < count; i++) {
        device_allocator_[i]->releaseAndFreeEvents();
    }
}

bool SecondaryMemoryAllocator::checkBlockIsSafe(const c10::DataPtr &ptr) {
    if (!ptr.get()) {
        return true;
    }
    /*
    if (ptr.get_deleter() != &local_raw_delete) {
        return true;
    }*/
    device::DeviceBlock *block = get_allocated_block(ptr.get());
    ZBCCL_ASSERT_S(block != nullptr, "No allocated block can be found", Z_INVALID_PTR);
    return block->is_safe_;
}

void SecondaryMemoryAllocator::markAllBlockUnsafe(int device) {
    return device_allocator_[device]->markAllBlockUnsafe();
}

void SecondaryMemoryAllocator::updateBlockToSafe(const c10::DataPtr &ptr) {
    if (!ptr.get()) {
        return;
    }
    /*
    if (ptr.get_deleter() != &local_raw_delete) {
        return;
    }*/
    device::DeviceBlock *block = get_allocated_block(ptr.get());
    ZBCCL_ASSERT_S(block != nullptr, "No allocated block can be found", Z_INVALID_PTR);
    if (!block->is_safe_) {
        ZBCCL_LOG_INFO("Triggers to refresh the data of the unsafe memory block and remove the unsafe flag");
    }
    block->is_safe_ = true;
}

void SecondaryMemoryAllocator::assertValidDevice(int device) {
    const auto device_num = device_allocator_.size();
    ZBCCL_CHECK_S(0 <= device && device < static_cast<int64_t>(device_num), "Invalid device argument ", device,
                  ": did you call init?");
}

ZResult SecondaryMemoryAllocator::Initialize(zbccl_allocator_options_t *options, int32_t device_count) noexcept {
    int size = static_cast<int>(device_allocator_.size());
    if (size < device_count) {
        device_allocator_.resize(device_count);
        for (auto i = size; i < device_count; ++i) {
            device_allocator_[i] = std::make_unique<device::DeviceSMACachingAllocator>();
        }
    }
    return Z_OK;
}

ZResult SecondaryMemoryAllocator::Allocate(void **devPtr, int device, size_t size, aclrtStream stream) noexcept {
    assertValidDevice(device);
    device::DeviceBlock *block = device_allocator_[device]->malloc(device, size, stream);

    add_allocated_block(block);
    *devPtr = static_cast<void *>(block->ptr_);

    return Z_OK;
}

ZResult SecondaryMemoryAllocator::Free(void *ptr) noexcept {
    if (!ptr) {
        return Z_INVALID_PTR;
    }
    device::DeviceBlock *block = get_allocated_block(ptr, true);
    if (!block) {
        ZBCCL_LOG_WARN("invalid device pointer: " << ptr);
    }
    device_allocator_[block->deviceId_]->free(block);

    return Z_OK;
}

ZResult SecondaryMemoryAllocator::EmptyCache(bool check_error) {
    ZBCCL_LOG_DEBUG("Begin empty cache with check_error = " << check_error);
    int32_t current_device = 0;
    if (check_error) {
        ZBCCL_CHECK_S(c10_npu::GetDevice(&current_device) == ACL_SUCCESS, Z_RT_ERROR);
    } else {
        c10_npu::GetDevice(&current_device);
    }

    int device_count = static_cast<int>(device_allocator_.size());
    for (int device_idx = 0; device_idx < device_count; device_idx++) {
        if (check_error)
            ZBCCL_CHECK_S(c10_npu::SetDevice(device_idx) == ACL_SUCCESS, Z_RT_ERROR);
        else
            c10_npu::SetDevice(device_idx);
        device_allocator_[device_idx]->emptyCache(device_idx, check_error);
    }
    if (check_error) {
        ZBCCL_CHECK_S(c10_npu::MaybeSetDevice(current_device) == ACL_SUCCESS, Z_RT_ERROR);
    } else {
        c10_npu::MaybeSetDevice(current_device);
    }
    ZBCCL_LOG_DEBUG("End empty cache with check_error = " << check_error);

    return Z_OK;
}

ZResult SecondaryMemoryAllocator::RecordStream(void *ptr, c10_npu::NPUStream stream) {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr) {
        return Z_ERROR;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when NPU tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    // if (ptr.get_deleter() != &local_raw_delete) {
    //     return;
    // }

    device::DeviceBlock *block = get_allocated_block(ptr);
    // block must not be null reaching here
    ZBCCL_ASSERT_S(block != nullptr, "No allocated block can be found");
    device_allocator_[block->deviceId_]->recordStream(block, stream);

    return Z_OK;
}

ZResult SecondaryMemoryAllocator::EraseStream(void *ptr, c10_npu::NPUStream stream)
{
    if (!ptr) {
        return Z_ERROR;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when NPU tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    // if (ptr.get_deleter() != &local_raw_delete) {
    //     // TORCH_NPU_WARN_ONCE("Tensor not is not allocated by DirectMemoryAllocator, skip eraseStream.");
    //     return;
    // }

    device::DeviceBlock *block = get_allocated_block(ptr);
    if (!block) {
        ZBCCL_LOG_ERROR("invalid device pointer: " << ptr);
    }

    if (block->stream_ != c10_npu::getCurrentNPUStream(block->deviceId_).stream(false)) {
        // If the Stream applying for tensor block different from
        // the stream of submiting event wait task in HCCL synchronize()
        // method, the recordSteam can not be erased.
        // New tensor creation may use the block before HCCL op is complete.
        return Z_ERROR;
    }

    device_allocator_[block->deviceId_]->eraseStream(block, stream);

    return Z_OK;
}

ZResult SecondaryMemoryAllocator::BeginAllocateToPool(int device, c10_npu::MempoolId_t mempool_id, std::function<bool(aclrtStream)> filter) {
    assertValidDevice(device);
    device_allocator_[device]->beginAllocateToPool(mempool_id, filter);
    return Z_OK;
}

ZResult SecondaryMemoryAllocator::EndAllocateToPool(int device, c10_npu::MempoolId_t mempool_id) {
    assertValidDevice(device);
    device_allocator_[device]->endAllocateToPool(mempool_id);
    return Z_OK;
}

ZResult SecondaryMemoryAllocator::ReleasePool(int device, c10_npu::MempoolId_t mempool_id) {
    assertValidDevice(device);
    device_allocator_[device]->releasePool(mempool_id);
    return Z_OK;
}

}  // namespace sma
}  // namespace zbccl

extern "C" {
ZBCCL_API void *sma_malloc(size_t size, int device, aclrtStream stream) {
    void *ptr = nullptr;
    if (size == 0) {
        return ptr;
    }
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->Allocate(&ptr, device, size, stream);
    return ptr;
}

ZBCCL_API void sma_free(void *ptr, size_t size, int device, aclrtStream stream) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->Free(ptr);
}

ZBCCL_API void sma_init(int device_count) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->Initialize(nullptr, device_count);
}

ZBCCL_API void sma_empty_cache(bool check_error) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->EmptyCache(check_error);
}

ZBCCL_API void sma_record_stream(void *ptr, c10_npu::NPUStream stream) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->RecordStream(ptr, stream);
}

ZBCCL_API void sma_erase_stream(void *ptr, c10_npu::NPUStream stream) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->EraseStream(ptr, stream);
}

ZBCCL_API void sma_begin_allocate_to_pool(int device, c10_npu::MempoolId_t mempool_id, std::function<bool(aclrtStream)> filter) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->BeginAllocateToPool(device, mempool_id, filter);
}

ZBCCL_API void sma_end_allocate_to_pool(int device, c10_npu::MempoolId_t mempool_id) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->EndAllocateToPool(device, mempool_id);
}

ZBCCL_API void sma_release_pool(int device, c10_npu::MempoolId_t mempool_id) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->ReleasePool(device, mempool_id);
}

ZBCCL_API void *sma_get_base_addr(int device) {
    int device_i = 0;
    if (device < 0)
        c10_npu::GetDevice(&device_i);
    else
        device_i = device;
    return zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device_i]->shmem_base_addr_;
}

aclshmemx_uniqueid_t sma_default_flag_uid;
static char sma_g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};

int sma_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                 aclshmemx_init_attr_t *attributes)
{
    ZBCCL_ASSERT_RETURN(local_mem_size <= ACLSHMEM_MAX_LOCAL_SIZE, ACLSHMEM_INVALID_VALUE);
    ZBCCL_ASSERT_RETURN(n_pes <= ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    ZBCCL_ASSERT_RETURN(my_pe < ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(sma_g_ipport) - 1);

        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            //SHM_LOG_ERROR("my_pe:" << my_pe << " ip_port is nullptr!");
            return ACLSHMEM_INVALID_VALUE;
        }
    } else {
        //SHM_LOG_WARN("init with my_pe:" << my_pe << " ip_port is nullptr!");
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT,
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&sma_default_flag_uid);
    aclshmemx_uniqueid_t *uid_args = (aclshmemx_uniqueid_t *)(attributes->comm_args);
    uid_args->my_pe = my_pe;
    uid_args->n_pes = n_pes;
    return shmem_error_code_t::ACLSHMEM_SUCCESS;
}

ZBCCL_API void sma_init_shmem(int my_rank, int n_ranks, uint64_t local_mem_size, uint64_t meta_size, const char *ip_port) {
    std::cout << "sma init: " << my_rank << " " << n_ranks << " " << local_mem_size << " " << meta_size << " "
              << ip_port << std::endl;
    if (shmem_init_status() != 2) {
        auto status = shmem_set_conf_store_tls(false, nullptr, 0);
        ZBCCL_ASSERT_S(status == shmem_error_code_t::ACLSHMEM_SUCCESS, "[E]shmem shmem_set_conf_store_tls error.");
        shmem_init_attr_t attributes;
        sma_set_attr(my_rank, n_ranks, local_mem_size, ip_port, &attributes);
        status = shmem_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
        ZBCCL_ASSERT_S(status == shmem_error_code_t::ACLSHMEM_SUCCESS, "[E]shmem shmem_init_attr error.");
    }

    int device = 0;
    c10_npu::GetDevice(&device);

    if (!zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device]->mem_heap_inited_) {
        void *shmem_base_addr_ = shmem_malloc(local_mem_size);
        zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device]->mem_heap_inited_ = true;
        zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device]->mem_heap_pool_ =
                std::make_shared<zbccl::sma::heap::MemoryHeap>(shmem_base_addr_ + meta_size,
                                                               local_mem_size - meta_size);
        zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device]->shmem_base_addr_ = shmem_base_addr_;
    }
}

ZBCCL_API void sma_init_heap(void *base_ptr, uint64_t local_mem_size) {
    int device = 0;
    c10_npu::GetDevice(&device);

    if (!zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device]->mem_heap_inited_) {
        void *shmem_base_addr_ = base_ptr;
        //ZBCCL_CHECK_S(!is_simulation, "[E]sma currently do not support simulation on this init.");
        zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device]->mem_heap_inited_ = true;
        zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device]->mem_heap_pool_ =
                std::make_shared<zbccl::sma::heap::MemoryHeap>(shmem_base_addr_, local_mem_size);
        zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device]->shmem_base_addr_ = shmem_base_addr_;
    }
    else {
        ZBCCL_LOG_WARN("re-entrance into sma init, skip this time init");
    }
}

}