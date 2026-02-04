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
#include <sstream>
#include <stdexcept>
#include <string>
#include "zbccl_sma_config.h"

namespace zbccl {
namespace sma {

void SMAConfig::parseEnv(const char *env)
{
    // If empty, set the default values
    max_split_size_ = std::numeric_limits<size_t>::max();
    garbage_collection_threshold_ = 0;

    if (env == nullptr) {
        return;
    }

    std::vector<std::string> config;
    lexArgs(env, config);

    for (size_t i = 0; i < config.size(); i++) {
        if (config[i].compare(MaxSplitSizeMB) == 0) {
            i = parseMaxSplitSize(config, i);
        } else if (config[i].compare(GarbageCollectionThreshold) == 0) {
            i = parseGarbageCollectionThreshold(config, i);
        //} else if (config[i] == BaseAddrAlignedKB) {
        //    i = parseAddrAlignSize(config, i);
        } else if (config[i] == UseSMAAllocator) {
            i = parseUseSMAAllocator(config, i);
        } else if (config[i] == UseVMMForStaticMemory) {
            i = parseUseVMMForStaticMemory(config, i);
        } else if (config[i] == SmallHeapSize) {
            i = parseSmallHeapSize(config, i);
        } else if (config[i] == SmallHeapThreshold) {
            i = parseSmallHeapThresHold(config, i);
        } else if (config[i] == SegmentSizeMB) {
            i = parseSegmentSizeMb(config, i);
        } else {
            ZBCCL_CHECK_S(false, "Unrecognized SMAConfig option: ", config[i]);
        }

        if (i + 1 < config.size()) {
            consumeToken(config, ++i, ',');
        }
    }
}

void SMAConfig::lexArgs(const char *env, std::vector<std::string> &config)
{
    std::vector<char> buf;

    size_t env_length = strlen(env);
    for (size_t i = 0; i < env_length; i++) {
        if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
            if (!buf.empty()) {
                config.emplace_back(buf.begin(), buf.end());
                buf.clear();
            }
            config.emplace_back(1, env[i]);
        } else if (env[i] != ' ') {
            buf.emplace_back(static_cast<char>(env[i]));
        }
    }
    if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
    }
}

void SMAConfig::consumeToken(const std::vector<std::string> &config, size_t i, const char c)
{
    ZBCCL_CHECK_S(i < config.size() && config[i].compare(std::string(1, c)) == 0,
        "Error parsing SMAConfig settings, expected ", c);
}

size_t SMAConfig::parseMaxSplitSize(const std::vector<std::string> &config, size_t i)
{
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        size_t val1 = static_cast<size_t>(stoi(config[i]));
        ZBCCL_CHECK_S(val1 > kLargeBuffer / (1024 * 1024),
                    "SMAConfig option max_split_size_mb too small, must be > ", kLargeBuffer / (1024 * 1024));
        val1 = std::max(val1, kLargeBuffer / (1024 * 1024));
        val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
        max_split_size_ = val1 * 1024 * 1024;
    } else {
        ZBCCL_CHECK_S(false, "Error, expecting max_split_size_mb value");
    }
    return i;
}

size_t SMAConfig::parseGarbageCollectionThreshold(const std::vector<std::string> &config, size_t i)
{
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        double val1 = stod(config[i]);
        ZBCCL_CHECK_S(val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0");
        ZBCCL_CHECK_S(val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0");
        garbage_collection_threshold_ = val1;
    } else {
        ZBCCL_CHECK_S(false, "Error, expecting garbage_collection_threshold value");
    }
    return i;
}

/*
size_t SMAConfig::parseAddrAlignSize(const std::vector<std::string> &config, size_t i)
{
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        size_t val = static_cast<size_t>(stoi(config[i]));
        ZBCCL_CHECK_S(config[i].length() == std::to_string(val).length(),
                    "SMAConfig option base_addr_aligned_kb error, must be [0~16], dtype is int");
        // ZBCCL_CHECK_S(val >= 0, "SMAConfig option base_addr_aligned_kb error, must be [0~16], dtype is int");
        ZBCCL_CHECK_S(val <= kAlignRoundLarge / 1024,
                    "SMAConfig option base_addr_aligned_kb error, must be [0~16], dtype is int");
        base_addr_aligned_size_ = val * 1024;
    } else {
        ZBCCL_CHECK_S(false, "Error, expecting base_addr_aligned_kb value");
    }
    return i;
}*/

size_t SMAConfig::parseSegmentSizeMb(const std::vector<std::string> &config, size_t i)
{
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        size_t val = static_cast<size_t>(stoi(config[i]));
        segment_size_mb_ = val * kMB;
    } else {
        ZBCCL_CHECK_S(false, "Error, expecting segment_size_mb value");
    }
    return i;
}

size_t SMAConfig::parseUseSMAAllocator(const std::vector<std::string> &config, size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        ZBCCL_CHECK_S(i < config.size() && (config[i] == "True" || config[i] == "False"),
                    "Expected a single True/False argument for use_sma_allocator");
        use_sma_allocator_ = (config[i] == "True");
    } else {
        ZBCCL_CHECK_S(false, "Error, expecting use_sma_allocator value");
    }
    return i;
}

size_t SMAConfig::parseUseVMMForStaticMemory(const std::vector<std::string> &config, size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        ZBCCL_CHECK_S(i < config.size() && (config[i] == "True" || config[i] == "False"),
                      "Expected a single True/False argument for use_vmm_for_static_memory");
        use_vmm_for_static_memory_ = (config[i] == "True");
    } else {
        ZBCCL_CHECK_S(false, "Error, expecting use_vmm_for_static_memory value");
    }
    return i;
}

size_t SMAConfig::parseSmallHeapSize(const std::vector<std::string> &config, size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        size_t val = static_cast<size_t>(stoi(config[i]));
        small_heap_size_ = val;
    } else {
        ZBCCL_CHECK_S(false, "Error, expecting small_heap_size value");
    }
    return i;
}

size_t SMAConfig::parseSmallHeapThresHold(const std::vector<std::string> &config, size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        size_t val = static_cast<size_t>(stoi(config[i]));
        small_heap_threshold_ = val;
    } else {
        ZBCCL_CHECK_S(false, "Error, expecting small_heap_threshold value");
    }
    return i;
}

}  // namespace sma
}  // namespace zbccl
