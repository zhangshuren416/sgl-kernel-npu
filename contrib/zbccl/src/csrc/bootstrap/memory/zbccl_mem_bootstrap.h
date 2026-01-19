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
namespace bootstrap {
enum MemBoostrapType : uint16_t {
    MBT_MEMFABRIC = 0,
    MBT_ACLSHMEM,

    MBT_BUTT
};

class MemBootstrap;
using MemBootstrapPtr = ZRef<MemBootstrap>;

struct MemBootstrapOptions {
    MemBoostrapType boostrapType = MBT_MEMFABRIC; /* memory init type */
    uint16_t deviceId = 0;                        /* device id */
    uint32_t rankCount = 0;                       /* total rank count */
    uint32_t rankId = 0;                          /* my rank id */
    uint64_t totalMemSize = 0;                    /* total memory size */
    uint32_t flags = 0;                           /* optional flags */
    uint32_t dataOperationType = 0;               /* data operation type MTE etc */
    std::string ipPort;                           /* SHM exchange ip port*/

    friend std::ostream &operator<<(std::ostream &os, const MemBootstrapOptions &options)
    {
        os << "MemBootstrapOptions [boostrapType: " << options.boostrapType << ", deviceId: " << options.deviceId
           << ", rankCount: " << options.rankCount << ", rankId: " << options.rankId
           << ", totalMemSize: " << options.totalMemSize << ", flags: " << options.flags
           << ", dataOperationType: " << options.dataOperationType << ", ipPort: " << options.ipPort << "]";

        return os;
    }
};

struct MemBootstrapOutput {
    void *gvaDevice = nullptr;
    void *myGvaDevice = nullptr;
    uint64_t memorySizeDevice = 0;

    friend std::ostream &operator<<(std::ostream &os, const MemBootstrapOutput &output)
    {
        os << "MemBootstrapOutput [deviceGva: " << output.gvaDevice << ", myDeviceGva: " << output.myGvaDevice
           << ", memorySizeDevice: " << output.memorySizeDevice << "]";

        return os;
    }
};

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
