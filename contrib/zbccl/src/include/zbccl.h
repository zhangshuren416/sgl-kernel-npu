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
#ifndef ZBCCL_H_
#define ZBCCL_H_

#include "zbccl_mem_allocator.h"
#include "zbccl_operations.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get version of zbccl
 *
 * @return string of version
 */
const char *zbccl_version();

/**
 * @brief Bootstrap zbccl
 *
 * @param options          [in] options of bootstrap
 * @param flags            [in] optional flags
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
void zbccl_unboostrap(uint32_t flags);

/**
 * @brief Set external log function
 *
 * @param func             [in] logger function
 *
 * @return 0 if successful
 */
int32_t zbccl_set_logger(void (*func)(int, const char *));

/**
 * @brief Set logger level
 *
 * @param level            [in] level, 0:debug 1:info 2:warn 3:error
 *
 * @return 0 if successful
 */
int32_t zbccl_set_logger_level(int level);

/**
 * @brief Get last error message if have
 *
 * @return error message, empty string if no error
 */
const char* zbccl_get_last_error_msg();

/**
 * @brief Get and clear last error message
 *
 * @return error message, empty string if no error
 */
const char* zbccl_get_and_clear_last_error_msg();

#ifdef __cplusplus
}
#endif

#endif  // ZBCCL_H_
