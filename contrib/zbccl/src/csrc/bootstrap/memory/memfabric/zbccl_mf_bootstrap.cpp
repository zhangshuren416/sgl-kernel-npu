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

namespace zbccl {
namespace bootstrap {
const char *DIR_PATH_OF_MemFabric_LIBRARY = "";

using namespace underapi;

ZResult MemFabricBoostrap::Initialize() noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (initialized_) {
        ZBCCL_LOG_INFO("MemFabric bootstrap already initialized, no action required");
        return Z_OK;
    }

    /* verify basic options */
    auto result = MemBootstrap::VerifyOptions();
    if (result != Z_OK) {
        return result;
    }

    /* dl open mf library which should be placed under LD_LIBRARY_PATH */
    result = DlMfApi::LoadLibrary(DIR_PATH_OF_MemFabric_LIBRARY);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("Load library of MemFabric failed @ " << DIR_PATH_OF_MemFabric_LIBRARY
                                                              << ", error: " << DlMfApi::SmemGetLastErrMsg());
        return Z_LOAD_BOOTSTRAP_LIBRARY_FAILED;
    }

    /* init MemFabric */
    DlMfApi::SmemSetExternLogger(OutLogger::Instance().GetExternalLogFunction());
    result = DlMfApi::SmemInit(0);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("Initialize MemFabric failed, error: " << DlMfApi::SmemGetLastErrMsg());
        DlMfApi::CleanupLibrary();
        return Z_INIT_BOOTSTRAP_FAILED;
    }

    /* create SHM of MemFabric */
    DlMfApi::SmemShmConfigInit(&shmConfig_);
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
    output_.myGvaDevice =
        reinterpret_cast<void *>(reinterpret_cast<uint64_t>(deviceGva) + options_.rankId * options_.totalMemSize);
    output_.memorySizeDevice = options_.totalMemSize;

    initialized_ = true;

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