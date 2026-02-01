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
#include "zbccl_bootstrap_default.h"

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
    ZBCCL_VALIDATE_RETURN(result == Z_OK, "create bootstrap object failed, initialization failed", nullptr);

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

BootstrapPtr Bootstrap::Get()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gBootstrap != nullptr) {
        return gBootstrap;
    }

    ZBCCL_LOG_WARN("Get bootstrap failed as it is not created");
    return nullptr;
}

ZResult Bootstrap::VerifyOptions() noexcept
{
    ZBCCL_VALIDATE_RETURN(0 <= options_.btType && options_.btType < BOOT_BY_BUTT,
                          "invalid option, bootstrapType is invalid", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(options_.ipPort != nullptr, "invalid option, ipPort is nullptr", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(options_.worldSize <= RANK_COUNT_MAX_LIMIT, "invalid option, worldSize too large",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(options_.rankId < options_.worldSize, "invalid option, rankId should be less than worldSize",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(options_.deviceId < DEVICE_COUNT_MAX_LIMIT, "invalid option, deviceId is incorrect",
                          Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(options_.deviceMemorySize < MEMORY_SIZE_CAP, "invalid options, memory size is too large",
                          Z_INVALID_PARAM);

    if (static_cast<uint64_t>(options_.cclGroupCap) * options_.cclMetaSpaceSize * 1024 >= options_.deviceMemorySize) {
        ZBCCL_LOG_ERROR("total meta space size is GE total device memory size.");
        return Z_INVALID_PARAM;
    }

    if (options_.cclMetaSpaceSize * 1024 <= OPERATE_PARAM_SIZE) {
        ZBCCL_LOG_ERROR("single meta space size is too small");
        return Z_INVALID_PARAM;
    }
    return Z_OK;
}

ZResult Bootstrap::Initialize() noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (inited_) {
        ZBCCL_LOG_INFO("Bootstrap already initialized, no action required");
        return Z_OK;
    }

    /* verify basic options */
    auto result = VerifyOptions();
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("Initialize bootstrap options invalid, ret=" << result);
        return result;
    }

    /* create memory bootstrap, before create comm bootstrap */
    result = CreateMemBootstrap();
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("Create mem bootstrap failed. ret=" << result);
        return result;
    }

    ZBCCL_LOG_DEBUG("bootstrap init success.");
    inited_ = true;
    return Z_OK;
}

void Bootstrap::UnInitialize() noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!inited_) {
        ZBCCL_LOG_INFO("Bootstrap not initialized");
        return;
    }

    /* destroy memory bootstrap */
    DestroyMemoryBootstrap();

    /* set flag */
    inited_ = false;
}

ZResult Bootstrap::CreateMemBootstrap() noexcept
{
    /* translate options from api options to internal options */
    MemBootstrapOptions memOptions{};
    memOptions.boostrapType = static_cast<MemBoostrapType>(options_.btType);
    memOptions.deviceId = options_.deviceId;
    memOptions.rankCount = options_.worldSize;
    memOptions.rankId = options_.rankId;
    memOptions.totalMemSize = options_.deviceMemorySize;
    memOptions.flags = options_.flags;
    memOptions.dataOperationType = options_.dataOperationType;
    memOptions.ipPort = std::string(options_.ipPort);

    ZBCCL_LOG_INFO("MemBootstrapOptions dump: " << memOptions);

    /* new bootstrap */
    auto memBootstrap = MemBootstrap::Create(memOptions);
    if (memBootstrap == nullptr) {
        ZBCCL_LOG_ERROR("Create MemoryBootstrap instance failed");
        return Z_ERROR;
    }

    /* initialize */
    auto result = memBootstrap->Initialize();
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("mem bootstrap init failed, ret=" << result);
        return result;
    }

    void *deviceGva;                    /* gva of the world */
    uint64_t allocatedDeviceMemorySize; /* actually allocated memory size */
    void *myDeviceGva;                  /* gva of this rank */
    void *myCCLMetaDeviceGva;           /* gva of ccl meta of this rank */
    uint64_t metaSizeOfDevice;          /* size of device memory for SMA */
    void *mySMAGva;                     /* gva of sma of this rank */
    uint64_t smaSizeOfDevice;           /* size of device memory for SMA */

    /* assign output */
    auto &memOutput = memBootstrap->GetOutput();
    output_.deviceGva = memOutput.gvaDevice;
    output_.myDeviceGva = memOutput.myGvaDevice;
    output_.allocatedDeviceMemorySize = memOutput.memorySizeDevice;
    output_.myCCLMetaDeviceGva = memOutput.myGvaDevice;
    output_.metaSizeOfDevice = options_.cclMetaSpaceSize;
    /* translate to bytes */
    output_.metaSizeOfDevice = output_.metaSizeOfDevice * 1024 * options_.cclGroupCap;
    output_.mySMAGva =
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(output_.myDeviceGva) + output_.metaSizeOfDevice);
    output_.smaSizeOfDevice = output_.allocatedDeviceMemorySize - output_.metaSizeOfDevice;

    memBootstrap_ = memBootstrap;
    ZBCCL_LOG_DEBUG("create mem bootstrap success.");
    return Z_OK;
}

void Bootstrap::DestroyMemoryBootstrap() noexcept
{
    if (memBootstrap_ == nullptr) {
        return;
    }

    /* swap to tmp bootstrap */
    auto tmpBootstrap = memBootstrap_;
    memBootstrap_ = nullptr;

    /* do un-initialize */
    tmpBootstrap->UnInitialize();
}

ZResult Bootstrap::AcquireCommGroupId(uint32_t max, uint32_t &uniqueId) noexcept
{
    if (memBootstrap_ == nullptr) {
        ZBCCL_LOG_DEBUG("Not bootstrapped");
        return Z_NOT_BOOTSTRAPPED;
    }

    return memBootstrap_->AcquireCommGroupId(max, uniqueId);
}

void Bootstrap::ReleaseCommGroupId(uint32_t uniqueId) noexcept
{
    if (memBootstrap_ == nullptr) {
        ZBCCL_LOG_DEBUG("Not bootstrapped");
        return;
    }

    memBootstrap_->ReleaseCommGroupId(uniqueId);
}
}  // namespace bootstrap
}  // namespace zbccl