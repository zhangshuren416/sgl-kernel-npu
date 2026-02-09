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
#ifndef ZBCCL_COMM_STRUCT_H
#define ZBCCL_COMM_STRUCT_H

#include <string>
#include <vector>

#define ZBCCL_CYCLE_PROFILING_SIZE  (64 * 1024L)
#define ZBCCL_MAX_AIV_SIZE_PER_NPU  48
#define ZBCCL_MAX_AIC_SIZE_PER_NPU  24
#define ZBCCL_PROFILING_FRAMES_MAX  (ZBCCL_CYCLE_PROFILING_SIZE / 2 / sizeof(int64_t) / \
                                    (ZBCCL_MAX_AIV_SIZE_PER_NPU + ZBCCL_MAX_AIC_SIZE_PER_NPU))
#define ZBCCL_SCALAR_CACHELINE_SIZE 64
#define ZBCCL_AIV_MAX_EXP_NUM       6                 // ceil(log2(48))
#define ZBCCL_CORE_BARRIER_SIZE     (ZBCCL_MAX_AIV_SIZE_PER_NPU * ZBCCL_AIV_MAX_EXP_NUM * ZBCCL_SCALAR_CACHELINE_SIZE)
#define ZBCCL_U64_CACHELINE_SIZE    (ZBCCL_SCALAR_CACHELINE_SIZE / sizeof(uint64_t))
#define ZBCCL_PROFILING_PRINT_WIDTH 20

struct zbccl_profiling_block_t {
    int64_t ccount[ZBCCL_MAX_AIC_SIZE_PER_NPU][ZBCCL_PROFILING_FRAMES_MAX];
    int64_t ccycle[ZBCCL_MAX_AIC_SIZE_PER_NPU][ZBCCL_PROFILING_FRAMES_MAX];
    int64_t vcount[ZBCCL_MAX_AIV_SIZE_PER_NPU][ZBCCL_PROFILING_FRAMES_MAX];
    int64_t vcycle[ZBCCL_MAX_AIV_SIZE_PER_NPU][ZBCCL_PROFILING_FRAMES_MAX];
};

enum zbccl_profiling_name_t : uint16_t {
    ZBCCL_PROF_BARRIER = 0,
    ZBCCL_PROF_ALLGATHER_KERNEL_ALL = 1,
    ZBCCL_PROF_BUTT = 2,                // should less than or equal ZBCCL_PROFILING_FRAMES_MAX
};

// keep name len <= ZBCCL_PROFILING_PRINT_WIDTH
const std::vector<std::string> g_profName = {
    "BARRIER",
    "AG_KERNEL_ALL",
};

#define ZBCCL_C_PROF_START(comm, frameId)                                                           \
    if (AscendC::GetBlockIdx() < ZBCCL_MAX_AIC_SIZE_PER_NPU && (frameId) < ZBCCL_PROF_BUTT) {       \
        auto block = reinterpret_cast<__gm__ zbccl_profiling_block_t *>(comm->profilingGva);        \
        PipeBarrier<PIPE_ALL>();                                                                    \
        auto cycles = AscendC::GetSystemCycle();                                                    \
        block->ccycle[AscendC::GetBlockIdx()][(frameId)] -= cycles;                                 \
    }

#define ZBCCL_C_PROF_STOP(comm, frameId)                                                            \
    if (AscendC::GetBlockIdx() < ZBCCL_MAX_AIC_SIZE_PER_NPU && (frameId) < ZBCCL_PROF_BUTT)  {      \
        auto block = reinterpret_cast<__gm__ zbccl_profiling_block_t *>(comm->profilingGva);        \
        PipeBarrier<PIPE_ALL>();                                                                    \
        auto cycles = AscendC::GetSystemCycle();                                                    \
        block->ccycle[AscendC::GetBlockIdx()][(frameId)] += cycles;                                 \
        block->ccount[AscendC::GetBlockIdx()][(frameId)] += 1;                                      \
    }

#define ZBCCL_V_PROF_START(comm, frameId)                                                           \
    if (AscendC::GetBlockIdx() < ZBCCL_MAX_AIV_SIZE_PER_NPU && (frameId) < ZBCCL_PROF_BUTT) {       \
        auto block = reinterpret_cast<__gm__ zbccl_profiling_block_t *>(comm->profilingGva);        \
        PipeBarrier<PIPE_ALL>();                                                                    \
        auto cycles = AscendC::GetSystemCycle();                                                    \
        block->vcycle[AscendC::GetBlockIdx()][frameId] -= cycles;                                   \
    }


#define ZBCCL_V_PROF_STOP(comm, frameId)                                                            \
    if (AscendC::GetBlockIdx() < ZBCCL_MAX_AIC_SIZE_PER_NPU && (frameId) < ZBCCL_PROF_BUTT) {       \
        auto block = reinterpret_cast<__gm__ zbccl_profiling_block_t *>(comm->profilingGva);        \
        PipeBarrier<PIPE_ALL>();                                                                    \
        auto cycles = AscendC::GetSystemCycle();                                                    \
        block->vcycle[AscendC::GetBlockIdx()][(frameId)] += cycles;                                 \
        block->vcount[AscendC::GetBlockIdx()][(frameId)] += 1;                                      \
    }

/**
 * @brief group info of this communicator, this struct will be copy to device, keep it simple
 */
struct CommGroupInfo {
    uint16_t groupSize = 0;                                 /* the ranks in the group */
    uint16_t myGroupRank = 0;                               /* rank id in the group */
    uintptr_t myMetaGva = 0;                                /* gva of mine */
    uintptr_t myParamDataGva = 0;                           /* gva of for param exchange of operation */
    uintptr_t myAddressExchangeGva = 0;                     /* gva of for address exchange of operation */
    uint64_t sizeForCommGroupInfo = 0;                      /* max memory size of passing param from host to device */
    uint64_t sizeForParam = 0;                              /* max memory size of exchange param */
    uint64_t sizeForExchangeAddress = 0;                    /* max memory size for exchange operation data addresses */
    uint64_t fftsConfig;                                    /* copy from CommGroupOptions.fftsConfig */
    uint16_t peerGroupRank2WorldRank[ZBCCL_MAX_RANKS] = {}; /* rank id in group to world rank id relationship */
    uint64_t localDeviceMemSize;                            /* copy from CommGroupOptions.localDeviceMemSize */
    uintptr_t profilingGva;                                 /* gva of profiling data */
    uint64_t sizeOfProfiling;                               /* max memory size of profiling data */
    uint64_t vecCounter[ZBCCL_U64_CACHELINE_SIZE];          /* vector counter space */
    uint64_t vecBarrier[ZBCCL_U64_CACHELINE_SIZE];          /* vector barrier space */
    uint8_t coreCounter[ZBCCL_CORE_BARRIER_SIZE];           /* core counter space */
    uint8_t coreBarrier[ZBCCL_CORE_BARRIER_SIZE];           /* core barrier space */
};

#endif  // ZBCCL_COMM_STRUCT_H
