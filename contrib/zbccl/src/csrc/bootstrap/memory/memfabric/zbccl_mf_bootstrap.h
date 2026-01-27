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
#ifndef ZBCCL_MF_BOOTSTRAP_H
#define ZBCCL_MF_BOOTSTRAP_H

#include "smem_shm_def.h"
#include "zbccl_mem_bootstrap.h"

namespace zbccl {
namespace bootstrap {
class MemFabricBoostrap : public MemBootstrap
{
public:
    MemFabricBoostrap(const MemBootstrapOptions &options) : MemBootstrap(options) {}
    ~MemFabricBoostrap() override
    {
        UnInitialize();
    }

    ZResult Initialize() noexcept override;
    void UnInitialize() noexcept override;

private:
    ZResult GetMemFabricLibPath(std::string &path) noexcept;
    ZResult GetAscendLibPath(std::string &path) noexcept;
    ZResult InitPreCheck() noexcept;
    ZResult CreateSHMSpace() noexcept;

private:
    /* one bootstrap maps to one shm handle */
    smem_shm_config_t shmConfig_;
    uint32_t shmId_ = 0;
    smem_shm_t shmHandle_ = nullptr;
    void *shmGva = nullptr;
};
}  // namespace bootstrap
}  // namespace zbccl

#endif  // ZBCCL_MF_BOOTSTRAP_H
