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
#include "zbccl_mem_bootstrap.h"
#include "zbccl_mf_bootstrap.h"

namespace zbccl {
namespace bootstrap {
MemBootstrapPtr MemBootstrap::Create(const MemBootstrapOptions &options)
{
    if (options.boostrapType == MemBoostrapType::MBT_MEMFABRIC) {
        auto bootstrap = ZMakeRef<MemFabricBoostrap>(options);
        if (bootstrap == nullptr) {
            ZBCCL_LOG_ERROR("Create MemFabric bootstrap failed, probably out of memory");
            return nullptr;
        }
        return bootstrap.Get();
    } else if (options.boostrapType == MemBoostrapType::MBT_ACLSHMEM) {
        ZBCCL_LOG_ERROR("Create ACLSHMEM bootstrap failed, as it is not supported");
        return nullptr;
    }

    return nullptr;
}

ZResult MemBootstrap::VerifyOptions()
{
    // TODO
    return Z_OK;
}
}  // namespace bootstrap
}  // namespace zbccl