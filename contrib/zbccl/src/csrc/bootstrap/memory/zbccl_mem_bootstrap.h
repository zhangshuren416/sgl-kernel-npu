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
#include "zbccl_bootstrap_types.h"

namespace zbccl {
namespace bootstrap {
class MemBootstrap : public ZReferable
{
public:
    static MemBootstrapPtr Create(const MemBootstrapOptions &options);

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
     * @brief Get output after initialized
     *
     * @return output
     */
    const MemBootstrapOutput &GetOutput() const;

    /**
     * @brief Get an unique id for communicator (i.e. a group)
     * which should start from 0
     *
     * @param max          [in] max number that the 'uniqueId' could be
     * @param uniqueId     [in/out] id acquired
     *
     * @return 0 if successful
     */
    virtual ZResult AcquireCommGroupId(uint32_t max, uint32_t &uniqueId) noexcept = 0;

    /**
     * @brief Release the id acquired by AcquireCommGroupId() function
     *
     * @param uniqueId     [in] the id to be released
     */
    virtual void ReleaseCommGroupId(uint32_t uniqueId) noexcept = 0;

protected:
    MemBootstrap(const MemBootstrapOptions &options) : options_(options) {}
    ZResult VerifyOptions();

protected:
    bool initialized_ = false;
    std::mutex mutex_;

    MemBootstrapOptions options_;
    MemBootstrapOutput output_;
};

inline const MemBootstrapOutput &MemBootstrap::GetOutput() const
{
    return output_;
}
}  // namespace bootstrap
}  // namespace zbccl

#endif  // ZBCCL_MEM_BOOTSTRAP_H
