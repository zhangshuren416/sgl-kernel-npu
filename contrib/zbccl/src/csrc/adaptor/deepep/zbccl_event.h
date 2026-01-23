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
#ifndef ZBCCL_EVENT_H_
#define ZBCCL_EVENT_H_

#include <memory>
#include "zbccl_logger.h"

namespace zbccl {
namespace adaptor {
namespace deep_ep {

struct EventHandle {
    EventHandle() {}

    EventHandle(const EventHandle &other) = default;

    void current_stream_wait() const
    {
        return;
    }
};

}  // namespace deep_ep
}  // namespace adaptor
}  // namespace zbccl
#endif  // ZBCCL_EVENT_H_