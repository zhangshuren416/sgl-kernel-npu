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

#include <pybind11/pybind11.h>

#include "c10_npu_dma.h"
#include "zbccl_pytorch_process_group.h"

namespace py = pybind11;

PYBIND11_MODULE(zbccl, m) {
    m.doc() = "zbccl package";

    auto allocator = m.def_submodule("allocator", "zbccl allocator");
    auto npu_adaptor = m.def_submodule("npu_adaptor", "zbccl npu pytorch adaptor");

    pybind11_allocator(allocator);
    pybind11_adaptor(npu_adaptor);
}