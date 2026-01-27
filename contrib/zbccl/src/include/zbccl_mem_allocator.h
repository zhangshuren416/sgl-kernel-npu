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
#ifndef ZBCCL_MEM_ALLOCATOR_H_
#define ZBCCL_MEM_ALLOCATOR_H_

#include "zbccl_def.h"
#include "zbccl_defines.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the allocator with options, this is allocator is secondary memory allocator,
 * the original memory is already allocated from Device, for example from mem fabric or shmem on Ascend.
 * The allocator can be plugged into torch, any address allocated from this allocator can be accessed by
 * other device directly, then we don't need to have CCL buffer to store the data temporarily.
 * this action must be called before pluggable allocator automatic use zbccl_pluggable_init.
 *
 * @param options          [in] the options of allocator
 * @param flags            [in] optional flags
 * @return 0 if successful
 */
int32_t zbccl_sma_init(zbccl_allocator_options_t *options, int32_t flags);

/**
 * @brief Un-initialize allocator
 *
 * @param flags            [in] optional flags
 */
void zbccl_sma_uninit(int32_t flags);


/**
 * @brief Initialize API of official torch pluggable memory allocator
 *
 * @param device_count     [in] number of devices
 * @return memory ptr is successful, nullptr is failed
 */
void zbccl_pluggable_init(int32_t device_count);

/**
 * @brief Allocate memory API of official torch pluggable memory allocator
 *
 * @param size             [in] size of memory to be allocated
 * @param device           [in] device id
 * @param stream           [in] current stream
 * @return memory ptr is successful, nullptr is failed
 */
void *zbccl_pluggable_malloc(size_t size, int32_t device, aclrtStream stream);

/**
 * @brief Free memory API of official torch pluggable memory allocator
 *
 * @param ptr              [in] pointer allocated by zbccl_torch_malloc
 * @param size             [in] size of memory
 * @param device           [in] device id
 * @param stream           [in] stream
 */
void zbccl_pluggable_free(void *ptr, size_t size, int32_t device, aclrtStream stream);

/**
 * @brief Empty cache API of official torch pluggable memory allocator
 *
 * @param check_error      [in] whether check on error allocate
 */
void zbccl_pluggable_empty_cache(bool check_error);


// TODO add adaptor for stream API since we dont want to include c10_npu in top level
// void zbccl_pluggable_record_stream(void *ptr, c10_npu::NPUStream stream);
// void zbccl_pluggable_erase_stream(void *ptr, c10_npu::NPUStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ZBCCL_MEM_ALLOCATOR_H_
