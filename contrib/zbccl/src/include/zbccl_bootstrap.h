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
#ifndef ZBCCL_BOOTSTRAP_H
#define ZBCCL_BOOTSTRAP_H

#include "zbccl_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
  * @brief Initialize 'zbccl_bootstrap_options_t' with default values
 *
 * @param options          [in] options to be initialized
 *
 * @return 0 if successful
 */
int32_t zbccl_bootstrap_options_init(zbccl_bootstrap_options_t *options);

/**
 * @brief Bootstrap zbccl
 *
 * @param options          [in] options of bootstrap
 * @param output           [out] bootstrap info after work done
 *
 * @return 0 if successful
 */
int32_t zbccl_bootstrap(zbccl_bootstrap_options_t *options, zbccl_bootstrap_output_t* output);

/**
 * @brief Un-bootstrap zbccl
 *
 * @param flags            [in] optional flags
 */
int32_t zbccl_unboostrap(uint32_t flags);

#ifdef __cplusplus
}
#endif


#endif  // ZBCCL_BOOTSTRAP_H
