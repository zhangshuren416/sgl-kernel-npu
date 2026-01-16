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
#ifndef ZBCCL_MEM_BOOTSTRAP_H
#define ZBCCL_MEM_BOOTSTRAP_H

#include "zbccl_common_includes.h"

namespace zbccl {
namespace underapi {
enum MemBoostrapType {
    MBT_MEMFABRIC = 0,
    MBT_ACLSHMEM,

    MBT_BUTT
};

class MemBootstrap;
using MemBootstrapPtr = ZRef<MemBootstrap>;

struct MemBootstrapOptions {
    MemBoostrapType boostrapType = MBT_MEMFABRIC; /* memory init type */
    uint32_t rankCount = 0;                       /* total rank count */
    uint32_t rankId = 0;                          /* my rank id */
    uint64_t totalMemSize = 0;                    /* total memory size */
    uint32_t flags = 0;                           /* optional flags */
};

class MemBootstrap : public ZReferable
{
public:
    MemBootstrapPtr Create(const MemBootstrapOptions &options);

public:
    ~MemBootstrap() override = default;

    /**
     * @brief Do real initialization according to bootstrap options, including
     * a) create shmem object
     * b) setup gva
     *
     * @return 0 if successful
     */
    virtual ZResult Initialize() noexcept = 0;

    /**
     * @brief Un-initialization
     */
    virtual void UnInitialize() noexcept = 0;

    /**
     * @brief Get the local part of GVA for secondary memory allocation,
     * this should be called after initialization
     *
     * @return nullptr is not initialized successfully, loca GVA if initialized successfully
     */
    virtual void *GetMyGVA() noexcept = 0;

protected:
    MemBootstrap(const MemBootstrapOptions &options) : options_(options) {}
    ZResult VerifyOptions();

protected:
    bool initialized_ = false;
    std::mutex mutex_;

    MemBootstrapOptions options_;
};
}  // namespace underapi
}  // namespace zbccl

#endif  // ZBCCL_MEM_BOOTSTRAP_H
