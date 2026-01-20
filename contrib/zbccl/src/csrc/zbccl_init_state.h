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
#ifndef ZBCCL_INIT_STATE_H
#define ZBCCL_INIT_STATE_H

#include "zbccl_common_includes.h"

namespace zbccl {
struct ZBCCLInitStateExt {
    zbccl_bootstrap_type_t btType;      /* bootstrap type */
    uint16_t worldSize = 0;             /* world size*/
    uint16_t worldRankId = 0;           /* world rank id*/
    uint16_t deviceId = 0;              /* device id */
    uint16_t cclMetaSpaceSize;          /* optional, in KB, default 1MB, min: 512KB, max: 4MB */
    uint16_t cclGroupCap;               /* optional, max count of ccl Group, default 128, min: 1, max: 512*/
    void *gvaDevice = nullptr;          /* global gva */
    void *myCCLMetaDeviceGva = nullptr; /* gva of ccl meta of this rank */
    uint64_t metaSizeOfDevice = 0;      /* size of device memory for SMA */
    void *mySMAGva = nullptr;           /* gva of sma of this rank */
    uint64_t smaSizeOfDevice = 0;       /* size of device memory for SMA */
};

class ZBCCLInitState
{
public:
    static ZBCCLInitState &Instance()
    {
        static ZBCCLInitState gInitState;
        return gInitState;
    }

public:
    ZBCCLInitState() = default;
    ~ZBCCLInitState() = default;

    void Bootstrapped(bool bootstrapped) noexcept;
    bool Bootstrapped() const noexcept;

    bool HasCommunicator() const noexcept;
    void CommunicatorCreated(uint16_t count = 1) noexcept;
    void CommunicatorDestroy(uint16_t count = 1) noexcept;

    void SmaInitialized(bool smaInited) noexcept;
    bool SmaInitialized() const noexcept;

    void Reset() noexcept;

public:
    ZBCCLInitStateExt ext_{};

private:
    std::atomic<bool> bootstrapped_{false};
    std::atomic<bool> smaInited_{false};
    std::atomic<int16_t> communicatorCount_{0};
};

inline void ZBCCLInitState::Bootstrapped(bool bootstrapped) noexcept
{
    bootstrapped_ = bootstrapped;
}

inline bool ZBCCLInitState::Bootstrapped() const noexcept
{
    return bootstrapped_.load();
}

inline bool ZBCCLInitState::HasCommunicator() const noexcept
{
    return communicatorCount_.load() > 0;
}

inline void ZBCCLInitState::CommunicatorCreated(uint16_t count) noexcept
{
    communicatorCount_ += count;
}
inline void ZBCCLInitState::CommunicatorDestroy(uint16_t count) noexcept
{
    communicatorCount_ -= count;
}

inline void ZBCCLInitState::SmaInitialized(bool smaInited) noexcept
{
    smaInited_ = smaInited;
}

inline bool ZBCCLInitState::SmaInitialized() const noexcept
{
    return smaInited_.load();
}

inline void ZBCCLInitState::Reset() noexcept
{
    bootstrapped_ = false;
    smaInited_ = false;
    communicatorCount_ = 0;
    bzero(&ext_, sizeof(ZBCCLInitStateExt));
}

}  // namespace zbccl

#endif  // ZBCCL_INIT_STATE_H
