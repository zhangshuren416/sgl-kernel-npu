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
#ifndef ZBCCL_SMA_CONFIG_H
#define ZBCCL_SMA_CONFIG_H

#include "zbccl_sma_common.h"
#include "zbccl_common_includes.h"

namespace zbccl {
namespace sma {

class SMAConfig
{
public:
    static size_t max_split_size()
    {
        return instance().max_split_size_;
    }

    static double garbage_collection_threshold()
    {
        return instance().garbage_collection_threshold_;
    }

    /*
    static size_t base_addr_aligned_size()
    {
        return instance().base_addr_aligned_size_;
    }*/

    static size_t segment_size_mb()
    {
        return instance().segment_size_mb_;
    }

    static SMAConfig &instance() noexcept
    {
        static SMAConfig *s_instance = ([]() {
            auto inst = new SMAConfig();
            const char *env = getenv(kPytorchNPUAllocConf);
            inst->parseEnv(env);
            return inst;
        })();
        return *s_instance;
    }

private:
    size_t max_split_size_;
    double garbage_collection_threshold_;
    // size_t base_addr_aligned_size_ = kAlignRoundLarge;
    size_t segment_size_mb_;

    SMAConfig():
        max_split_size_(std::numeric_limits<size_t>::max()),
        garbage_collection_threshold_(0),
        // base_addr_aligned_size_(kAlignRoundLarge),
        segment_size_mb_(0)
    {}

    void parseEnv(const char *env);

    void lexArgs(const char *env, std::vector<std::string> &config);
    void consumeToken(const std::vector<std::string> &config, size_t i, const char c);
    size_t parseMaxSplitSize(const std::vector<std::string> &config, size_t i);
    size_t parseGarbageCollectionThreshold(const std::vector<std::string> &config, size_t i);
    // size_t parseAddrAlignSize(const std::vector<std::string> &config, size_t i);
    size_t parseSegmentSizeMb(const std::vector<std::string> &config, size_t i);
};
}  // namespace sma
}  // namespace zbccl

#endif  // ZBCCL_SMA_CONFIG_H
