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
#include <dlfcn.h>

#include "dl_mf_api.h"

namespace zbccl {
namespace underapi {
std::mutex DlMfApi::gMutex;
bool DlMfApi::gLoaded = false;
void *gMfSmemHandle = nullptr;
const char *gMfLibName = "libmf_smem.so";

mfSmemInitFunc DlMfApi::gMfSmemInit = nullptr;
mfSmemCreateConfigStoreFunc DlMfApi::gMfSmemCreateConfigStore = nullptr;
mfSmemSetExternLoggerFunc DlMfApi::gMfSmemSetExternLogger = nullptr;
mfSmemSetLogLevelFunc DlMfApi::gMfSmemSetLogLevel = nullptr;
mfSmemUnInitFunc DlMfApi::gMfSmemUnInit = nullptr;
mfSmemGetLastErrMsgFunc DlMfApi::gMfSmemGetLastErrMsg = nullptr;
mfSmemGetAndClearErrMsgFunc DlMfApi::gMfSmemGetAndClearErrMsg = nullptr;

mfSmemShmConfigInitFunc DlMfApi::gMfSmemShmConfigInit = nullptr;
mfSmemShmInitFunc DlMfApi::gMfSmemShmInit = nullptr;
mfSmemShmUnInitFunc DlMfApi::gMfSmemShmUnInit = nullptr;
mfSmemShmQuerySupportDataOperationFunc DlMfApi::gMfSmemShmQuerySupportDataOperation = nullptr;
mfSmemCreateFunc DlMfApi::gMfSmemCreate = nullptr;
mfSmemShmDestroyFunc DlMfApi::gMfSmemShmDestroy = nullptr;
mfSmemShmSetExtraContextFunc DlMfApi::gMmfSmemShmSetExtraContext = nullptr;
mfSmemShmGetGlobalRankFunc DlMfApi::gMfSmemShmGetGlobalRank = nullptr;
mfSmemShmGetGlobalRankSizeFunc DlMfApi::gMfSmemShmGetGlobalRankSize = nullptr;
mfSmemShmControlBarrierFunc DlMfApi::gMfSmemShmControlBarrier = nullptr;
mfSmemShmControlAllGatherFunc DlMfApi::gMfSmemShmControlAllGather = nullptr;
mfSmemShmTopologyCanReachFunc DlMfApi::gMfSmemShmTopologyCanReach = nullptr;
mfSmemShmRegisterExitFunc DlMfApi::gMfSmemShmRegisterExit = nullptr;
mfSmemShmGlobalExitFunc DlMfApi::gMfSmemShmGlobalExit = nullptr;

ZResult DlMfApi::LoadLibrary(const std::string &libDirPath)
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return Z_OK;
    }

    std::string realPath;
    if (!Func::LibraryRealPath(libDirPath, std::string(gMfLibName), realPath)) {
        ZBCCL_LOG_ERROR(libDirPath << "get real library [" << gMfLibName << "] failed");
        return Z_FILE_NOT_FOUND;
    }

    /* dlopen library */
    gMfSmemHandle = dlopen(realPath.c_str(), RTLD_NOW | RTLD_NODELETE);
    if (gMfSmemHandle == nullptr) {
        ZBCCL_LOG_ERROR("Failed to open library [" << realPath << "], error: " << dlerror());
        return Z_DL_OPEN_LIB_FAILED;
    }

    /* load sym */
    DL_LOAD_SYM(gMfSmemInit, mfSmemInitFunc, gMfSmemHandle, "smem_init");
    DL_LOAD_SYM(gMfSmemCreateConfigStore, mfSmemCreateConfigStoreFunc, gMfSmemHandle, "smem_create_config_store");
    DL_LOAD_SYM(gMfSmemSetExternLogger, mfSmemSetExternLoggerFunc, gMfSmemHandle, "smem_set_extern_logger");
    DL_LOAD_SYM(gMfSmemSetLogLevel, mfSmemSetLogLevelFunc, gMfSmemHandle, "smem_set_log_level");
    DL_LOAD_SYM(gMfSmemUnInit, mfSmemUnInitFunc, gMfSmemHandle, "smem_uninit");
    DL_LOAD_SYM(gMfSmemGetLastErrMsg, mfSmemGetLastErrMsgFunc, gMfSmemHandle, "smem_get_last_err_msg");
    DL_LOAD_SYM(gMfSmemGetAndClearErrMsg, mfSmemGetAndClearErrMsgFunc, gMfSmemHandle,
                "smem_get_and_clear_last_err_msg");

    DL_LOAD_SYM(gMfSmemShmConfigInit, mfSmemShmConfigInitFunc, gMfSmemHandle, "smem_shm_config_init");
    DL_LOAD_SYM(gMfSmemShmInit, mfSmemShmInitFunc, gMfSmemHandle, "smem_shm_init");
    DL_LOAD_SYM(gMfSmemShmUnInit, mfSmemShmUnInitFunc, gMfSmemHandle, "smem_shm_uninit");
    DL_LOAD_SYM(gMfSmemShmQuerySupportDataOperation, mfSmemShmQuerySupportDataOperationFunc, gMfSmemHandle,
                "smem_shm_query_support_data_operation");
    DL_LOAD_SYM(gMfSmemCreate, mfSmemCreateFunc, gMfSmemHandle, "smem_shm_create");
    DL_LOAD_SYM(gMfSmemShmDestroy, mfSmemShmDestroyFunc, gMfSmemHandle, "smem_shm_destroy");
    DL_LOAD_SYM(gMmfSmemShmSetExtraContext, mfSmemShmSetExtraContextFunc, gMfSmemHandle, "smem_shm_set_extra_context");
    DL_LOAD_SYM(gMfSmemShmGetGlobalRank, mfSmemShmGetGlobalRankFunc, gMfSmemHandle, "smem_shm_get_global_rank");
    DL_LOAD_SYM(gMfSmemShmGetGlobalRankSize, mfSmemShmGetGlobalRankSizeFunc, gMfSmemHandle,
                "smem_shm_get_global_rank_size");
    DL_LOAD_SYM(gMfSmemShmControlBarrier, mfSmemShmControlBarrierFunc, gMfSmemHandle, "smem_shm_control_barrier");
    DL_LOAD_SYM(gMfSmemShmControlAllGather, mfSmemShmControlAllGatherFunc, gMfSmemHandle, "smem_shm_control_allgather");
    DL_LOAD_SYM(gMfSmemShmTopologyCanReach, mfSmemShmTopologyCanReachFunc, gMfSmemHandle,
                "smem_shm_topology_can_reach");
    DL_LOAD_SYM(gMfSmemShmRegisterExit, mfSmemShmRegisterExitFunc, gMfSmemHandle, "smem_shm_register_exit");
    DL_LOAD_SYM(gMfSmemShmGlobalExit, mfSmemShmGlobalExitFunc, gMfSmemHandle, "smem_shm_global_exit");

    gLoaded = true;
    return Z_OK;
}

void DlMfApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    gMfSmemInit = nullptr;
    gMfSmemCreateConfigStore = nullptr;
    gMfSmemSetExternLogger = nullptr;
    gMfSmemSetLogLevel = nullptr;
    gMfSmemUnInit = nullptr;
    gMfSmemGetLastErrMsg = nullptr;
    gMfSmemGetAndClearErrMsg = nullptr;

    gMfSmemShmConfigInit = nullptr;
    gMfSmemShmInit = nullptr;
    gMfSmemShmUnInit = nullptr;
    gMfSmemShmQuerySupportDataOperation = nullptr;
    gMfSmemCreate = nullptr;
    gMfSmemShmDestroy = nullptr;
    gMmfSmemShmSetExtraContext = nullptr;
    gMfSmemShmGetGlobalRank = nullptr;
    gMfSmemShmGetGlobalRankSize = nullptr;
    gMfSmemShmControlBarrier = nullptr;
    gMfSmemShmControlAllGather = nullptr;
    gMfSmemShmTopologyCanReach = nullptr;
    gMfSmemShmRegisterExit = nullptr;
    gMfSmemShmGlobalExit = nullptr;

    if (gMfSmemHandle != nullptr) {
        dlclose(gMfSmemHandle);
        gMfSmemHandle = nullptr;
    }
    gLoaded = false;
}
}  // namespace underapi
}  // namespace zbccl