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
#ifndef ZBCCL_BOOTSTRAP_DEFAULT_H
#define ZBCCL_BOOTSTRAP_DEFAULT_H

#include "zbccl_common_includes.h"
#include "zbccl_mem_bootstrap.h"

namespace zbccl {
namespace bootstrap {
class Bootstrap;
using BootstrapPtr = ZRef<Bootstrap>;

class Bootstrap : public ZReferable
{
public:
    static BootstrapPtr Create(const zbccl_bootstrap_options_t &options);
    static void Destroy();

public:
    explicit Bootstrap(const zbccl_bootstrap_options_t &options) : options_(options) {}
    ~Bootstrap() override
    {
        UnInitialize();
    }

    ZResult Initialize() noexcept;
    void UnInitialize() noexcept;

    const zbccl_bootstrap_output_t &GetOutput() const;

private:
    ZResult VerifyOptions() noexcept;

private:
    MemBootstrapPtr memBootstrap_{nullptr};
    zbccl_bootstrap_options_t options_;
    zbccl_bootstrap_output_t output_;

    std::mutex mutex_;
    bool inited_{false};

private:
    static std::mutex gMutex;
    static BootstrapPtr gBootstrap;
};

inline const zbccl_bootstrap_output_t &Bootstrap::GetOutput() const
{
    return output_;
}
}  // namespace bootstrap
}  // namespace zbccl

#endif  // ZBCCL_BOOTSTRAP_DEFAULT_H
