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
#ifndef ZBCCL_MEM_DEVICE_OPERATOR_H
#define ZBCCL_MEM_DEVICE_OPERATOR_H

namespace zbccl {
namespace bootstrap {
#if define(BOOTSTRAP_MEMFABRIC)
#elif define(BOOSTRAP_ACLSHMEM)
#endif
}  // namespace bootstrap
}  // namespace zbccl

#endif  // ZBCCL_MEM_DEVICE_OPERATOR_H
