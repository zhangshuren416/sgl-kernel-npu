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

    void WorldSize(uint16_t worldSize) noexcept;
    uint16_t WorldSize() const noexcept;

    void WorldRankId(uint16_t worldRank) noexcept;
    uint16_t WorldRankId() const noexcept;

    void DeviceId(uint16_t deviceId) noexcept;
    uint16_t DeviceId() const noexcept;

private:
    std::atomic<bool> bootstrapped_{false};
    std::atomic<bool> smaInited_{false};
    std::atomic<int16_t> communicatorCount_{false};
    std::atomic<uint16_t> worldSize_{0};
    uint16_t worldRankId_ = 0;
    uint16_t deviceId_ = 0;
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

inline void ZBCCLInitState::WorldSize(uint16_t worldSize) noexcept
{
    worldSize_ = worldSize;
}

inline uint16_t ZBCCLInitState::WorldSize() const noexcept
{
    return worldSize_.load();
}

inline void ZBCCLInitState::WorldRankId(uint16_t worldRank) noexcept
{
    worldRankId_ = worldRank;
}

inline uint16_t ZBCCLInitState::WorldRankId() const noexcept
{
    return worldRankId_;
}

inline void ZBCCLInitState::DeviceId(uint16_t deviceId) noexcept
{
    deviceId_ = deviceId;
}

inline uint16_t ZBCCLInitState::DeviceId() const noexcept
{
    return deviceId_;
}

}  // namespace zbccl

#endif  // ZBCCL_INIT_STATE_H
