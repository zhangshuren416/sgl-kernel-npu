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
#include "zbccl_mf_bootstrap.h"
#include "dl_mf_api.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace zbccl {
namespace bootstrap {

using namespace underapi;

ZResult MemFabricBoostrap::InitPrecheck() noexcept
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

    /* dl open mf library which should be placed under LD_LIBRARY_PATH */
    result = DlMfApi::LoadLibrary(memFabricLibPath);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("Load library of MemFabric failed @ " << memFabricLibPath
                                                              << ", error: " << DlMfApi::SmemGetLastErrMsg());
        return Z_LOAD_BOOTSTRAP_LIBRARY_FAILED;
    }
    return Z_OK;
}

ZResult MemFabricBoostrap::Initialize() noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);

    auto ret = InitPrecheck();
    if (ret != Z_OK) {
        return ret;
    }

    ret = InitMemfabric();
    if (ret != Z_OK) {
        DlMfApi::SmemUnInit();
        DlMfApi::CleanupLibrary();
        return ret;
    }

    ZBCCL_LOG_DEBUG("init mf bootstrap success.");
    return Z_OK;
}

ZResult MemFabricBoostrap::InitMemfabric() noexcept
{
    ZResult result = DlMfApi::SmemInit(0);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("smem init failed, ret is " << result);
        return result;
    }

    result = DlMfApi::SmemShmConfigInit(&shmConfig_);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("smem config init failed. ret is " << result);
        return result;
    }

    int32_t deviceId = -1;
    result = aclrtGetDevice(&deviceId);
    if (result != Z_OK || deviceId < 0) {
        ZBCCL_LOG_ERROR("get device id failed, ret is " << result);
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    result = DlMfApi::SmemShmInit(options_.ipPort.c_str(), options_.rankCount, options_.rankId, deviceId, &shmConfig_);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("smem shm init failed. ret is " << result);
        return result;
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
        ZBCCL_LOG_ERROR("Create shm by MemFabric failed, error: " << DlMfApi::SmemGetLastErrMsg());
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

    ZBCCL_LOG_DEBUG("init mem mf success.");
    initialized_ = true;
    return Z_OK;
}

int32_t MemFabricBoostrap::GetMemFabricLibPath(std::string &path) noexcept
{
    char *memFabricHome = std::getenv("MEMFABRIC_HYBRID_HOME_PATH");
    if (memFabricHome == nullptr) {
        ZBCCL_LOG_ERROR("get mem fabric home path empty.");
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    path = std::string(memFabricHome).append("/aarch64-linux/lib64/");
    if (!zbccl::Func::Realpath(path)) {
        ZBCCL_LOG_ERROR("input mem fabric lib path invalid");
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
}  // namespace bootstrap
}  // namespace zbccl