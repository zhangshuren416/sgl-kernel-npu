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

    /**
     * @brief Initialize bootstrap according to options
     *
     * @return 0 if successful, error if failed
     */
    ZResult Initialize() noexcept;

    /**
     * @brief Un-initialize bootstrap
     */
    void UnInitialize() noexcept;

    /**
     * @brief Get output if initialized
     *
     * @return output
     */
    const zbccl_bootstrap_output_t &GetOutput();

private:
    ZResult VerifyOptions() noexcept;
    ZResult CreateMemBootstrap() noexcept;
    void DestroyMemoryBootstrap() noexcept;

private:
    MemBootstrapPtr memBootstrap_{nullptr}; /* inner memory bootstrap */
    zbccl_bootstrap_options_t options_;     /* options from API */
    zbccl_bootstrap_output_t output_;       /* output for API */

    std::mutex mutex_;
    bool inited_{false};

private:
    static std::mutex gMutex;       /* lock for singleton boostrap */
    static BootstrapPtr gBootstrap; /* singleton bootstrap */
};

inline const zbccl_bootstrap_output_t &Bootstrap::GetOutput()
{
    std::lock_guard<std::mutex> guard(std::mutex);
    /* reset to if not initialized */
    if (!inited_) {
        bzero(&output_, sizeof(zbccl_bootstrap_output_t));
    }

    return output_;
}
}  // namespace bootstrap
}  // namespace zbccl

#endif  // ZBCCL_BOOTSTRAP_DEFAULT_H
