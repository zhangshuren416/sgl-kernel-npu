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
#include "zbccl_bootstrap.h"

namespace zbccl {
namespace bootstrap {
std::mutex Bootstrap::gMutex;
BootstrapPtr Bootstrap::gBootstrap = nullptr;

BootstrapPtr Bootstrap::Create(const zbccl_bootstrap_options_t &options)
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gBootstrap != nullptr) {
        return gBootstrap;
    }

    auto bootstrap = ZMakeRef<Bootstrap>(options);
    ZBCCL_VALIDATE_RETURN(bootstrap != nullptr, "create bootstrap object failed, probably out of memory", nullptr);

    auto result = bootstrap->Initialize();
    ZBCCL_VALIDATE_RETURN(result != Z_OK, "create bootstrap object failed, initialization failed", nullptr);

    gBootstrap = bootstrap;

    return gBootstrap;
}

void Bootstrap::Destroy()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gBootstrap == nullptr) {
        return;
    }

    gBootstrap->UnInitialize();
    gBootstrap = nullptr;
}

ZResult Bootstrap::VerifyOptions() noexcept
{
    // TODO
    return Z_OK;
}

ZResult Bootstrap::Initialize() noexcept
{
    // TODO
    return Z_OK;
}

void Bootstrap::UnInitialize() noexcept {}
}  // namespace bootstrap
}  // namespace zbccl