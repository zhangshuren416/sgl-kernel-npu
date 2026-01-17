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
#ifndef SGL_KERNEL_NPU_BAO_DL_MF_API_H
#define SGL_KERNEL_NPU_BAO_DL_MF_API_H

#include "zbccl_common_includes.h"
#include "smem_shm_def.h"

namespace zbccl {
namespace underapi {

using mfSmemInitFunc = int32_t (*)(uint32_t);
using mfSmemCreateConfigStoreFunc = int32_t (*)(const char *);
using mfSmemSetExternLoggerFunc = int32_t (*)(void (*func)(int level, const char *msg));
using mfSmemSetLogLevelFunc = int32_t (*)(int);
using mfSmemUnInitFunc = void (*)(void);
using mfSmemGetLastErrMsgFunc = const char *(*)(void);
using mfSmemGetAndClearErrMsgFunc = const char *(*)(void);

using mfSmemShmConfigInitFunc = int32_t (*)(smem_shm_config_t *);
using mfSmemShmInitFunc = int32_t (*)(const char *, uint32_t, uint32_t, uint16_t, smem_shm_config_t *);
using mfSmemShmUnInitFunc = void (*)(uint32_t);
using mfSmemShmQuerySupportDataOperationFunc = uint32_t (*)(void);
using mfSmemCreateFunc = smem_shm_t (*)(uint32_t, uint32_t, uint32_t, uint64_t, smem_shm_data_op_type, uint32_t,
                                        void **);
using mfSmemShmDestroyFunc = int32_t (*)(smem_shm_t, uint32_t);
using mfSmemShmSetExtraContextFunc = int32_t (*)(smem_shm_t, const void *, uint32_t);
using mfSmemShmGetGlobalRankFunc = uint32_t (*)(smem_shm_t);
using mfSmemShmGetGlobalRankSizeFunc = uint32_t (*)(smem_shm_t);
using mfSmemShmControlBarrierFunc = int32_t (*)(smem_shm_t);
using mfSmemShmControlAllGatherFunc = int32_t (*)(smem_shm_t, const char *, uint32_t, char *, uint32_t);
using mfSmemShmTopologyCanReachFunc = int32_t (*)(smem_shm_t, uint32_t, uint32_t *);
using mfSmemShmRegisterExitFunc = int32_t (*)(smem_shm_t, void (*exit)(int));
using mfSmemShmGlobalExitFunc = void (*)(smem_shm_t, int);

class DlMfApi
{
public:
    static ZResult LoadLibrary(const std::string &libDirPath);
    static void CleanupLibrary();

    DlMfApi() = delete;
    ~DlMfApi() = delete;

private:
    static std::mutex gMutex;
    static bool gLoaded;
    static void *gMfSmemHandle;
    static const char *gMfLibName;

    static mfSmemInitFunc gMfSmemInit;
    static mfSmemCreateConfigStoreFunc gMfSmemCreateConfigStore;
    static mfSmemSetExternLoggerFunc gMfSmemSetExternLogger;
    static mfSmemSetLogLevelFunc gMfSmemSetLogLevel;
    static mfSmemUnInitFunc gMfSmemUnInit;
    static mfSmemGetLastErrMsgFunc gMfSmemGetLastErrMsg;
    static mfSmemGetAndClearErrMsgFunc gMfSmemGetAndClearErrMsg;

    static mfSmemShmConfigInitFunc gMfSmemShmConfigInit;
    static mfSmemShmInitFunc gMfSmemShmInit;
    static mfSmemShmUnInitFunc gMfSmemShmUnInit;
    static mfSmemShmQuerySupportDataOperationFunc gMfSmemShmQuerySupportDataOperation;
    static mfSmemCreateFunc gMfSmemCreate;
    static mfSmemShmDestroyFunc gMfSmemShmDestroy;
    static mfSmemShmSetExtraContextFunc gMmfSmemShmSetExtraContext;
    static mfSmemShmGetGlobalRankFunc gMfSmemShmGetGlobalRank;
    static mfSmemShmGetGlobalRankSizeFunc gMfSmemShmGetGlobalRankSize;
    static mfSmemShmControlBarrierFunc gMfSmemShmControlBarrier;
    static mfSmemShmControlAllGatherFunc gMfSmemShmControlAllGather;
    static mfSmemShmTopologyCanReachFunc gMfSmemShmTopologyCanReach;
    static mfSmemShmRegisterExitFunc gMfSmemShmRegisterExit;
    static mfSmemShmGlobalExitFunc gMfSmemShmGlobalExit;
};
}  // namespace underapi
}  // namespace zbccl

#endif  // SGL_KERNEL_NPU_BAO_DL_MF_API_H
