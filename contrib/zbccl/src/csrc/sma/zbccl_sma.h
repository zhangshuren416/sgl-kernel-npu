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

#include "zbccl_sma_common.h"
#include "zbccl_sma_device.h"
#include "zbccl_sma_device_pool.h"

#include "shmem_api.h"  //  need include after zbccl_sma_common.h

namespace zbccl {
namespace sma {

class SecondaryMemoryAllocator : public ZReferable
{
private:
    std::mutex mutex_;

    // allocated blocks by device pointer
    ska::flat_hash_map<void *, device::DeviceBlock *> allocated_blocks_;

    void add_allocated_block(device::DeviceBlock *block);

    device::DeviceBlock *get_allocated_block(void *ptr, bool remove = false);

    bool initialized();

    void cleanEvent();

    // TODO fix those if pluggable also need
    bool checkBlockIsSafe(const c10::DataPtr &ptr);
    void markAllBlockUnsafe(int device);
    void updateBlockToSafe(const c10::DataPtr &ptr);

public:
    std::vector<std::unique_ptr<device::DeviceSMACachingAllocator>> device_allocator_;

    SecondaryMemoryAllocator();
    ~SecondaryMemoryAllocator() override = default;;

    static ZRef<SecondaryMemoryAllocator> GetInstance() {
        static ZRef<SecondaryMemoryAllocator> instance = new SecondaryMemoryAllocator();
        return instance;
    }

    /**
     * @brief Initialize the allocator
     *
     * @param options           [in] options for the allocator
     * @param device_cnt        [in] device count
     * @return 0 is successful
     */
    ZResult Initialize(zbccl_allocator_options_t *options, int32_t device_count) noexcept;

    /**
     * @brief Un-initialize the allocator
     *
     * @param flags        [in] extra flags
     */
    void UnInitialize(int32_t flags) noexcept;

    /**
     * @brief Allocate memory
     *
     * @param devPtr         [in] ptr of memory pointer wanted to be allocated
     * @param device         [in] device id
     * @param size           [in] allocate size
     * @param stream         [in] aclrtStream
     * @return 0 if successful
     */
    ZResult Allocate(void **devPtr, int device, size_t size, aclrtStream stream) noexcept;

    /**
     * @brief Free memory
     *
     * @param ptr          [in] memory pointer allocated by <i>Allocate</i>
     * @return 0 if successful
     */
    ZResult Free(void *ptr) noexcept;

    /**
     * @brief free cached memory
     *
     * @param check_error  [in] assert when aclrt func error
     * @return 0 if successful
     */
    ZResult EmptyCache(bool check_error);

    /**
     * @brief record a stream on its depend mem block
     *
     * @param ptr         [in] mem ptr of stream rely on
     * @param stream      [in] relied stream
     * @return 0 if successful
     */
    ZResult RecordStream(void *ptr, c10_npu::NPUStream stream);

    /**
     * @brief release a stream on its depend mem block
     *
     * @param ptr         [in] mem ptr of stream rely on
     * @param stream      [in] stream to be released
     * @return 0 if successful
     */
    ZResult EraseStream(void *ptr, c10_npu::NPUStream stream);
};
using SMAPtr = ZRef<SecondaryMemoryAllocator>;

}  // namespace sma
}  // namespace zbccl

extern "C" {
ZBCCL_API void *sma_malloc(size_t size, int device, aclrtStream stream);

ZBCCL_API void sma_init(int device_count);

ZBCCL_API void sma_empty_cache(bool check_error);

ZBCCL_API void sma_free(void *ptr, size_t size, int device, aclrtStream stream);

ZBCCL_API void sma_record_stream(void *ptr, c10_npu::NPUStream stream);

ZBCCL_API void sma_erase_stream(void *ptr, c10_npu::NPUStream stream);

ZBCCL_API void *sma_get_base_addr(int device = -1);

ZBCCL_API void sma_init_shmem(int my_rank, int n_ranks, uint64_t local_mem_size, uint64_t meta_size, const char *ip_port);
}

#endif  // ZBCCL_SMA_H

