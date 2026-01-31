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
using mfSmemShmCreateFunc = smem_shm_t (*)(uint32_t, uint32_t, uint32_t, uint64_t, smem_shm_data_op_type, uint32_t,
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

using mfSmemShmSubgroupBarrierFunc = int32_t (*)(smem_shm_t, const char *, uint32_t, uint32_t);
using mfSmemShmSubgroupAllGatherFunc = int32_t (*)(smem_shm_t, const char *, uint32_t, uint32_t, const char *, uint32_t,
                                                   char *, uint32_t);

class DlMfApi
{
public:
    static ZResult LoadLibrary(const std::string &libDirPath);
    static void CleanupLibrary();

    DlMfApi() = delete;
    ~DlMfApi() = delete;

    static ZResult SmemInit(uint32_t flags);
    static ZResult SmemSetExternLogger(void (*func)(int level, const char *msg));
    static ZResult SmemSetLoggerLevel(int level);
    static void SmemUnInit(void);
    static const char *SmemGetLastErrMsg(void);
    static const char *SmemGetAndClearLastErrMsg(void);

    /**
     * @brief Initialize smem_shm_config_t, i.e. set to default value
     *
     * @param config           [in] config to be initialized
     * @return 0 if successful
     */
    static ZResult SmemShmConfigInit(smem_shm_config_t *config);

    /**
     * @brief Initialize shm library with global config store
     * all processes need to call this function before creating a shm object,
     * all processes will connect to a global config store with specified ipPort,
     * this function will finish when all processes connected or timeout;
     * the global config store will be used to exchange information about shm object and team
     *
     * @param configStoreIpPort[in] ipPort of config store, e.g. tcp://ip:port or tcp://[ip]:port
     * @param worldSize        [in] size of processes
     * @param rankId           [in] local rank id in world size
     * @param deviceId         [in] device npu id
     * @param config           [in] config, see @smem_shm_config_t
     * @return 0 if successfully, negative value if failed, use @ref smem_get_last_error_msg to get last err msg
     */
    static ZResult SmemShmInit(const char *configStoreIpPort, uint32_t worldSize, uint32_t rankId, uint16_t deviceId,
                               smem_shm_config_t *config);

    /**
     * @brief Un-initialize shm library with destroy all things
     *
     * @param flags            [in] optional flags, set to 0
     */
    static void SmemShmUnInit(uint32_t flags);

    /**
     * @brief Query supported data operation type
     * @return the set of smem_shm_data_op_type
     */
    static uint32_t SmemShmQuerySupportDataOperation(void);

    /**
     * @brief Create shm object peer by peer
     *
     * @param id               [in] id of the shm object
     * @param rankSize         [in] rank count
     * @param rankId           [in] my rank id
     * @param symmetricSize    [in] local memory contributed to the shm object, all ranks must the same size
     * @param dataOpType       [in] data operation engine type, i.e. MTE, SDMA, RDMA etc
     * @param flags            [in] optional flags
     * @param gva              [out] global virtual address created, it can be passed to kernel to data operations
     * @return shm object created if successful, null if failed, use @ref smem_get_last_error_msg to get last error
     * message
     */
    static smem_shm_t SmemShmCreate(uint32_t id, uint32_t rankSize, uint32_t rankId, uint64_t symmetricSize,
                                    smem_shm_data_op_type dataOpType, uint32_t flags, void **gva);

    /**
     * @brief Destroy shm object
     *
     * @param handle           [in] the shm object to be destroyed
     * @param flags            [in] optional flags
     * @return 0 if successful
     */
    static ZResult SmemShmDestroy(smem_shm_t handle, uint32_t flags);

    /**
     * @brief Set user extra context of shm object
     *
     * @param handle           [in] the shm object to be set
     * @param context          [in] extra context ptr
     * @param size             [in] extra context size (max is 64K)
     * @return 0 if successful
     */
    static ZResult SmemShmSetExtraContext(smem_shm_t handle, const void *context, uint32_t size);

    /**
     * @brief Get local rank of a shm object
     *
     * @param handle           [in] the shm object
     * @return local rank in the input object, return UINT32_MAX if error
     */
    static uint32_t SmemShmGetGlobalRank(smem_shm_t handle);

    /**
     * @brief Get rank size of a shm object
     *
     * @param handle           [in] the shm object
     * @return rank size in the input object, return UINT32_MAX if error
     */
    static uint32_t SmemShmGetGlobalRankSize(smem_shm_t handle);

    /**
     * @brief Do barrier on a shm object, using control network
     *
     * @param handle           [in] the shm object
     * @return 0 if successful, other is error
     */
    static ZResult SmemShmControlBarrier(smem_shm_t handle);

    /**
     * @brief Do all gather on a shm object, using control network
     *
     * @param handle           [in] the shm object
     * @param sendBuf          [in] input data buf
     * @param sendSize         [in] input data buf size
     * @param recvBuf          [in] output data buf
     * @param recvSize         [in] output data buf size
     * @return 0 if successful
     */
    static ZResult SmemShmControlAllGather(smem_shm_t handle, const char *sendBuf, uint32_t sendSize, char *recvBuf,
                                           uint32_t recvSize);

    /**
     * @brief Query if remote rank can ranch
     *
     * @param handle           [in] shm object
     * @param remoteRank       [in] remote rank
     * @param reachInfo        [out] reach info, the set of smem_shm_data_op_type
     * @return 0 if successful
     */
    static ZResult SmemShmTopologyCanReach(smem_shm_t handle, uint32_t remoteRank, uint32_t *reachInfo);

    /**
     * @brief Register function of exit
     *
     * @param exit             [in] global exit option, every rank will apply this function
     *                              to complete global exit
     * @param handle           [in] shm object
     * @return 0 if successful
     */
    static ZResult SmemShmRegisterExit(smem_shm_t handle, void (*exit)(int));

    /**
     * @brief Wait for all ranks exit
     *
     * @param handle           [in] shm object
     * @param status           [in] int
     */
    static void SmemShmGlobalExit(smem_shm_t handle, int status);

    /**
     * @brief Do barrier operation with sub partition of world, there is no need to setup sub group firstly,
     * just need to make sure the key is following the rule:
     * a) key is a string
     * b) key should be the same for all participators (i.e. same in the sub group)
     * c) key should be different with other sub group, otherwise it will be messed up
     *
     * @param handle           [in] shm object
     * @param key              [in] key name for this barrier, which should be same in sub group but unique in the world
     * @param rankSize         [in] rank size of the sub group
     * @param rankId           [in] rank id in the sub group
     * @return 0 if successful
     */
    static ZResult SmemShmSubGroupBarrier(smem_shm_t handle, const std::string &key, uint32_t rankSize,
                                          uint32_t rankId);

    /**
     * @brief Do allGather operation with sub partition of world, there is no need to setup sub group firstly,
     * just need to make sure the key is following the rule:
     * a) key is a string
     * b) key should be the same for all participators (i.e. same in the sub group)
     * c) key should be different with other sub group, otherwise it will be messed up
     *
     * @param handle           [in] shm object
     * @param key              [in] key name for this barrier, which should be same in sub group but unique in the world
     * @param rankSize         [in] rank size of the sub group
     * @param rankId           [in] rank id in the sub group
     * @param sendBuf          [in] input data buf
     * @param sendSize         [in] input data buf size
     * @param recvBuf          [in] output data buf
     * @param recvSize         [in] output data buf size
     * @return
     */
    static ZResult SmemShmSubGroupAllGather(smem_shm_t handle, const std::string &key, uint32_t rankSize,
                                            uint32_t rankId, const char *sendBuf, uint32_t sendSize, char *recvBuf,
                                            uint32_t recvSize);

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
    static mfSmemShmCreateFunc gMfSmemShmCreate;
    static mfSmemShmDestroyFunc gMfSmemShmDestroy;
    static mfSmemShmSetExtraContextFunc gMfSmemShmSetExtraContext;
    static mfSmemShmGetGlobalRankFunc gMfSmemShmGetGlobalRank;
    static mfSmemShmGetGlobalRankSizeFunc gMfSmemShmGetGlobalRankSize;
    static mfSmemShmControlBarrierFunc gMfSmemShmControlBarrier;
    static mfSmemShmControlAllGatherFunc gMfSmemShmControlAllGather;
    static mfSmemShmTopologyCanReachFunc gMfSmemShmTopologyCanReach;
    static mfSmemShmRegisterExitFunc gMfSmemShmRegisterExit;
    static mfSmemShmGlobalExitFunc gMfSmemShmGlobalExit;

    static mfSmemShmSubgroupBarrierFunc gMfSmemShmSubgroupBarrier;
    static mfSmemShmSubgroupAllGatherFunc gMfSmemShmSubgroupAllGather;
};

inline ZResult DlMfApi::SmemInit(uint32_t flags)
{
    return gMfSmemInit(flags);
}

inline ZResult DlMfApi::SmemSetExternLogger(void (*func)(int level, const char *msg))
{
    return gMfSmemSetExternLogger(func);
}

inline ZResult DlMfApi::SmemSetLoggerLevel(int level)
{
    return gMfSmemSetLogLevel(level);
}

inline void DlMfApi::SmemUnInit(void)
{
    gMfSmemUnInit();
}

inline const char *DlMfApi::SmemGetLastErrMsg(void)
{
    return gMfSmemGetLastErrMsg();
}

inline const char *DlMfApi::SmemGetAndClearLastErrMsg(void)
{
    return gMfSmemGetAndClearErrMsg();
}

inline ZResult DlMfApi::SmemShmConfigInit(smem_shm_config_t *config)
{
    return gMfSmemShmConfigInit(config);
}

inline ZResult DlMfApi::SmemShmInit(const char *configStoreIpPort, uint32_t worldSize, uint32_t rankId,
                                    uint16_t deviceId, smem_shm_config_t *config)
{
    return gMfSmemShmInit(configStoreIpPort, worldSize, rankId, deviceId, config);
}

inline void DlMfApi::SmemShmUnInit(uint32_t flags)
{
    gMfSmemShmUnInit(flags);
}

inline uint32_t DlMfApi::SmemShmQuerySupportDataOperation(void)
{
    return gMfSmemShmQuerySupportDataOperation();
}

inline smem_shm_t DlMfApi::SmemShmCreate(uint32_t id, uint32_t rankSize, uint32_t rankId, uint64_t symmetricSize,
                                         smem_shm_data_op_type dataOpType, uint32_t flags, void **gva)
{
    return gMfSmemShmCreate(id, rankSize, rankId, symmetricSize, dataOpType, flags, gva);
}

inline ZResult DlMfApi::SmemShmDestroy(smem_shm_t handle, uint32_t flags)
{
    return gMfSmemShmDestroy(handle, flags);
}

inline ZResult DlMfApi::SmemShmSetExtraContext(smem_shm_t handle, const void *context, uint32_t size)
{
    return gMfSmemShmSetExtraContext(handle, context, size);
}

inline uint32_t DlMfApi::SmemShmGetGlobalRank(smem_shm_t handle)
{
    return gMfSmemShmGetGlobalRank(handle);
}

inline uint32_t DlMfApi::SmemShmGetGlobalRankSize(smem_shm_t handle)
{
    return gMfSmemShmGetGlobalRankSize(handle);
}

inline ZResult DlMfApi::SmemShmControlBarrier(smem_shm_t handle)
{
    return gMfSmemShmControlBarrier(handle);
}

inline ZResult DlMfApi::SmemShmControlAllGather(smem_shm_t handle, const char *sendBuf, uint32_t sendSize,
                                                char *recvBuf, uint32_t recvSize)
{
    return gMfSmemShmControlAllGather(handle, sendBuf, sendSize, recvBuf, recvSize);
}

inline ZResult DlMfApi::SmemShmTopologyCanReach(smem_shm_t handle, uint32_t remoteRank, uint32_t *reachInfo)
{
    return gMfSmemShmTopologyCanReach(handle, remoteRank, reachInfo);
}

inline ZResult DlMfApi::SmemShmRegisterExit(smem_shm_t handle, void (*exit)(int))
{
    return gMfSmemShmRegisterExit(handle, exit);
}

inline void DlMfApi::SmemShmGlobalExit(smem_shm_t handle, int status)
{
    gMfSmemShmGlobalExit(handle, status);
}

inline ZResult DlMfApi::SmemShmSubGroupBarrier(smem_shm_t handle, const std::string &key, uint32_t rankSize,
                                               uint32_t rankId)
{
    return gMfSmemShmSubgroupBarrier(handle, key.c_str(), rankSize, rankId);
}

inline ZResult DlMfApi::SmemShmSubGroupAllGather(smem_shm_t handle, const std::string &key, uint32_t rankSize,
                                                 uint32_t rankId, const char *sendBuf, uint32_t sendSize, char *recvBuf,
                                                 uint32_t recvSize)
{
    return gMfSmemShmSubgroupAllGather(handle, key.c_str(), rankSize, rankId, sendBuf, sendSize, recvBuf, recvSize);
}
}  // namespace underapi
}  // namespace zbccl

#endif  // SGL_KERNEL_NPU_BAO_DL_MF_API_H
