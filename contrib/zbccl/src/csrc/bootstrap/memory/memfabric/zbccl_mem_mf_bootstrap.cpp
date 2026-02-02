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
#include "zbccl_mem_mf_bootstrap.h"
#include "dl_mf_api.h"
#include "dl_cann_api.h"

namespace zbccl {
namespace bootstrap {

using namespace underapi;

ZResult MemFabricBoostrap::InitPreCheck() noexcept
{
    if (initialized_) {
        ZBCCL_LOG_INFO("MemFabric bootstrap already initialized, no action required");
        return Z_OK;
    }

    /* verify basic options */
    auto result = MemBootstrap::VerifyOptions();
    if (result != Z_OK) {
        return result;
    }

    std::string memFabricLibPath = "";
    result = GetMemFabricLibPath(memFabricLibPath);
    if (result != Z_OK || memFabricLibPath.empty()) {
        return result;
    }

    /* dl open mf library */
    result = DlMfApi::LoadLibrary(memFabricLibPath);
    if (result != Z_OK) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("Load library of MemFabric failed @ "
                                     << memFabricLibPath << ", error: " << DlMfApi::SmemGetLastErrMsg());
        return Z_LOAD_BOOTSTRAP_LIBRARY_FAILED;
    }

    std::string ascendLibPath = "";
    result = GetAscendLibPath(ascendLibPath);
    if (result != Z_OK || ascendLibPath.empty()) {
        return result;
    }

    /* dlopen ascend library */
    result = DlCannApi::LoadLibrary(ascendLibPath);
    if (result != Z_OK) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("Load library of ascendcl failed @ " << ascendLibPath << ", error: " << result);
        DlMfApi::CleanupLibrary();
        return Z_LOAD_BOOTSTRAP_LIBRARY_FAILED;
    }

    return Z_OK;
}

ZResult MemFabricBoostrap::Initialize() noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);

    auto result = InitPreCheck();
    if (result != Z_OK) {
        return result;
    }

    result = CreateSHMSpace();
    if (result != Z_OK) {
        DlMfApi::SmemUnInit();
        DlMfApi::CleanupLibrary();
        return result;
    }

    ZBCCL_LOG_DEBUG("Initialized mf bootstrap successfully");
    return Z_OK;
}

ZResult MemFabricBoostrap::CreateSHMSpace() noexcept
{
    auto result = DlMfApi::SmemInit(0);
    if (result != Z_OK) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("Call external api 'smem_init' failed, result: " << result);
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    result = DlMfApi::SmemShmConfigInit(&shmConfig_);
    if (result != Z_OK) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("Call external api 'smem_shm_config_init' failed, result: " << result);
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    int32_t deviceId = -1;
    result = underapi::DlCannApi::AclrtGetDevice(&deviceId);
    if (result != Z_OK || deviceId < 0) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("Call external api 'aclrtGetDevice' failed, result: " << result);
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    result = DlMfApi::SmemShmInit(options_.ipPort.c_str(), options_.rankCount, options_.rankId, deviceId, &shmConfig_);
    if (result != Z_OK) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("Call external api 'smem_shm_init' failed, result: " << result);
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    auto logger = OutLogger::Instance().GetExternalLogFunction();
    if (logger == nullptr) {
        logger = zbccl::OutLogger::DefaultLog;
    }
    DlMfApi::SmemSetExternLogger(logger);

    void *deviceGva = nullptr;
    auto tmpShmHandle = DlMfApi::SmemShmCreate(shmId_, options_.rankCount, options_.rankId, options_.totalMemSize,
                                               SMEMS_DATA_OP_MTE, options_.flags, &deviceGva);
    if (tmpShmHandle == nullptr) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("Call external api 'smem_shm_create' failed, result: " << result);
        DlMfApi::SmemUnInit();
        DlMfApi::CleanupLibrary();
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    /* get local GVA */
    shmHandle_ = tmpShmHandle;
    output_.gvaDevice = deviceGva;
    auto curGvaOffset = reinterpret_cast<uint64_t>(deviceGva) + options_.rankId * options_.totalMemSize;
    output_.myGvaDevice = reinterpret_cast<void *>(curGvaOffset);
    output_.memorySizeDevice = options_.totalMemSize;

    /* set logger level */
    auto loggerLevel = OutLogger::Instance().GetLogLevel();
    result = DlMfApi::SmemSetLoggerLevel(loggerLevel);
    if (result != Z_OK) {
        ZBCCL_LOG_WARN("Set logger level of MemFabric to " << loggerLevel << " failed, no action required");
    }

    ZBCCL_LOG_DEBUG("Initialized SHM space successfully");
    initialized_ = true;
    return Z_OK;
}

ZResult MemFabricBoostrap::GetMemFabricLibPath(std::string &path) noexcept
{
    char *memFabricLibPath = std::getenv("MEMFABRIC_HYBRID_LIBRARY_PATH");
    if (memFabricLibPath == nullptr) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("ENV MEMFABRIC_HYBRID_LIBRARY_PATH is not set, set this ENV properly");
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    path = std::string(memFabricLibPath);
    if (!zbccl::Func::Realpath(path)) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("Path with MEMFABRIC_HYBRID_LIBRARY_PATH is invalid");
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    return Z_OK;
}

ZResult MemFabricBoostrap::GetAscendLibPath(std::string &path) noexcept
{
    char *ascendHome = std::getenv("ASCEND_HOME_PATH");
    if (ascendHome == nullptr) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("ENV ASCEND_HOME_PATH is not set, set this ENV properly");
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    path = std::string(ascendHome).append("/lib64");
    if (!zbccl::Func::Realpath(path)) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("Path with ASCEND_HOME_PATH is invalid");
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    return Z_OK;
}

void MemFabricBoostrap::UnInitialize() noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_) {
        ZBCCL_LOG_INFO("MemFabric bootstrap not initialized, no action required");
        return;
    }

    /* close the SHM handle */
    if (shmHandle_ != nullptr) {
        DlMfApi::SmemShmDestroy(shmHandle_, 0);
        shmHandle_ = nullptr;
    }

    /* un-init MemFabric common stuff */
    DlMfApi::SmemUnInit();

    /* clean up library */
    DlMfApi::CleanupLibrary();

    initialized_ = false;
}

ZResult MemFabricBoostrap::AcquireCommGroupId(uint32_t max, uint32_t &uniqueId) noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_) {
        ZBCCL_LOG_INFO("MemFabric bootstrap not initialized, no action required");
        return Z_MEM_NOT_BOOTSTRAP;
    }

    return DlMfApi::SmemShmAtomicAllocValue(shmHandle_, max, &uniqueId);
}

ZResult MemFabricBoostrap::ReleaseCommGroupId(uint32_t uniqueId) noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_) {
        ZBCCL_LOG_INFO("MemFabric bootstrap not initialized, no action required");
        return Z_MEM_NOT_BOOTSTRAP;
    }

    return DlMfApi::SmemShmAtomicReleaseValue(shmHandle_, uniqueId);
}

ZResult MemFabricBoostrap::SubGroupAllGather(const std::string &key, uint32_t rankSize, uint32_t rankId,
                                             const char *sendBuf, uint32_t sendSize, char *recvBuf,
                                             uint32_t recvSize) noexcept
{
    ZBCCL_ASSERT_RETURN(!key.empty(), Z_INVALID_PARAM);
    ZBCCL_ASSERT_RETURN(rankSize > 0, Z_INVALID_PARAM);
    ZBCCL_ASSERT_RETURN(rankSize > rankId, Z_INVALID_PARAM);
    ZBCCL_ASSERT_RETURN(sendBuf != nullptr, Z_INVALID_PARAM);
    ZBCCL_ASSERT_RETURN(sendSize > 0, Z_INVALID_PARAM);
    ZBCCL_ASSERT_RETURN(recvBuf != nullptr, Z_INVALID_PARAM);
    ZBCCL_ASSERT_RETURN(recvSize > 0, Z_INVALID_PARAM);

    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_) {
        ZBCCL_LOG_INFO("MemFabric bootstrap not initialized, no action required");
        return Z_MEM_NOT_BOOTSTRAP;
    }

    ZBCCL_LOG_DEBUG("start sub group allGather, key: " << key << ", rankSize: " << rankSize << ", rankId: " << rankId
                                                       << ", sendSize: " << sendSize << ", recvSize: " << recvSize);

    return DlMfApi::SmemShmSubGroupAllGather(shmHandle_, key, rankSize, rankId, sendBuf, sendSize, recvBuf, recvSize);
}

ZResult MemFabricBoostrap::SetLoggerLevel(int level) noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_) {
        ZBCCL_LOG_INFO("MemFabric bootstrap not initialized, no action required");
        return Z_MEM_NOT_BOOTSTRAP;
    }

    ZBCCL_LOG_DEBUG("Try to set logger level of MemFabric to " << level);

    return DlMfApi::SmemSetLoggerLevel(level);
}
}  // namespace bootstrap
}  // namespace zbccl