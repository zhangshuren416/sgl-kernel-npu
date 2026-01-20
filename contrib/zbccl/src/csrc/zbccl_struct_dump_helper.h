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
#ifndef ZBCCL_STRUCT_DUMP_HELPER_H
#define ZBCCL_STRUCT_DUMP_HELPER_H

#include "zbccl_common_includes.h"

namespace zbccl {
static inline std::ostream &operator<<(std::ostream &os, const zbccl_bootstrap_options_t &options)
{
    os << "zbccl_bootstrap_options_t [flags: " << options.flags << ", bootstrap_type: " << options.btType
       << ", ipPort: " << ((options.ipPort == nullptr) ? "" : options.ipPort) << ", worldSize: " << options.worldSize
       << ", rankId: " << options.rankId << ", deviceId: " << options.deviceId
       << ", startConfigServer: " << options.startConfigServer << ", deviceMemorySize: " << options.deviceMemorySize
       << ", dataOperationType: " << options.dataOperationType << "]";

    return os;
}

static inline std::ostream &operator<<(std::ostream &os, const zbccl_bootstrap_output_t &output)
{
    os << "zbccl_bootstrap_output_t [deviceGva: " << output.deviceGva << ", myDeviceGva: " << output.myDeviceGva
       << ", allocatedDeviceMemorySize: " << output.allocatedDeviceMemorySize << "]";

    return os;
}

static inline std::ostream &operator<<(std::ostream &os, const zbccl_allocator_options_t &options)
{
    os << "zbccl_allocator_options_t [gva: " << options.gva << ", myGva: " << options.myGva
       << ", size: " << options.size << "]";

    return os;
}

static inline std::ostream &operator<<(std::ostream &os, const zbccl_ccl_options_t &options)
{
    os << "zbccl_ccl_options_t [backendType: " << options.backendType << ", flags: " << options.flags
       << ", isWorldGroup: " << options.isWorldGroup << ", groupSize: " << options.groupSize
       << ", groupRankId: " << options.groupRankId << ", isolateOpMeta: " << options.isolateOpMeta
       << ", deviceGva: " << options.deviceGva << ", myDeviceGva: " << options.myDeviceGva << "]";

    return os;
}
}  // namespace zbccl

#endif  // ZBCCL_STRUCT_DUMP_HELPER_H
