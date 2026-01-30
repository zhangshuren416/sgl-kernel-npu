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
#ifndef ZBCCL_OPERATIONS_H_
#define ZBCCL_OPERATIONS_H_

#include "zbccl_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create zero buffer communicator
 *
 * @param options              [in] communicator options
 * @param comm                 [out] created communicator
 * @return 0 if successful
 */
int32_t zbccl_comm_create(zbccl_comm_options_t *options, zbccl_comm_t *comm);

/**
 * @brief Get property of zero buffer communicator object
 *
 * @param comm                 [in] the communicator handle
 * @param property             [in/out]
 * @return 0 if successful
 */
int32_t zbccl_comm_get_property(zbccl_comm_t comm, zbccl_comm_property_t *property);

/**
 * @brief Get communicator object by name
 *
 * @param name                 [in] name of the communicator
 * @return comm object if successful, null if no such communicator
 */
zbccl_comm_t zbccl_comm_get_by_name(const char *name);

/**
 * @brief Destroy zero buffer communicator
 *
 * @param comm                 [in] the communicator to be destroyed
 * @param flags                [in] optional flags
 * @return
 */
int32_t zbccl_comm_destroy(zbccl_comm_t comm, uint32_t flags);

/**
 * @brief Do all reduce operation
 *
 * @param sendBuff             [in] pointer of send buffer
 * @param recvBuff             [in] pointer of receive buffer
 * @param count                [in] size of buffer
 * @param dataType             [in] data type
 * @param op                   [in] operation type of reduce
 * @param comm                 [in] zbccl communication handle
 * @param stream               [in] stream
 * @return 0 if successful
 */
int32_t zbccl_all_reduce(const void *send_buff, void *recv_buff, size_t count, zbccl_datatype_t data_type,
                         zbccl_reduce_op_t op, zbccl_comm_t comm, aclrtStream stream);

/**
 * @brief Do reduce scatter operation
 *
 * @param sendBuff             [in] pointer of send buffer
 * @param recvBuff             [in] pointer of receive buffer
 * @param recv_count           [in] size of buffer
 * @param dataType             [in] data type
 * @param op                   [in] operation type of reduce
 * @param comm                 [in] zbccl communication handle
 * @param stream               [in] stream
 * @return 0 if successful
 */
int32_t zbccl_reduce_scatter(const void *sendBuff, void *recvBuff, size_t recv_count, zbccl_datatype_t dataType,
                             zbccl_reduce_op_t op, zbccl_comm_t comm, aclrtStream stream);

/**
 * @brief Do all gather operation
 *
 * @param sendBuff             [in] pointer of send buffer
 * @param recvBuff             [in] pointer of receive buffer
 * @param send_count           [in] size of buffer
 * @param dataType             [in] data type
 * @param comm                 [in] zbccl communication handle
 * @param stream               [in] stream
 * @return 0 if successful
 */
int32_t zbccl_all_gather(const void *sendBuff, void *recvBuff, size_t send_count, zbccl_datatype_t dataType,
                         zbccl_comm_t comm, aclrtStream stream);

/**
 * @brief All2all operation
 *
 * @param sendBuff             [in]
 * @param recvBuff             [in]
 * @param data_count           [in]
 * @param dataType             [in]
 * @param stride_count         [in]
 * @param repeat               [in]
 * @param comm                 [in]
 * @param stream               [in]
 * @return
 */
int32_t zbccl_all_to_all(const void *sendBuff, void *recvBuff, uint64_t data_count, zbccl_datatype_t dataType,
                         uint64_t stride_count, uint8_t repeat, zbccl_comm_t comm, aclrtStream stream);

/**
 * @brief Calculate the number of tokens sent to each rank and other info
 *
 * @param sendTokensPerExpert  [in] the number of tokens to be sent to each expert
 * @param sendCount            [in] send data count
 * @param topKNum              [in] num of topK
 * @param recvBuff             [in] receive data
 * @param totalRecvTokens      [in/out] total recv token num
 * @param recvTokensPerExpert  [in/out] the number of tokens received by each expert
 * @param pushTargetOffset     [in/out] the token offset sent by different ranks to each expert
 * @param balanceMatrix        [in/out] the token range processed by each rank after balanced
 * @param comm                 [in] zbccl communication handle
 * @param stream               [in] stream
 * @param flags                [in] optional flags, reserved or extend
 * @return
 */
int32_t zbccl_dispatch_normal_notify(const zbccl_tensor_info_t *sendTokensPerExpert, int64_t sendCount, int64_t topKNum,
                                     const zbccl_tensor_info_t *recvBuff, int64_t *totalRecvTokens,
                                     const zbccl_tensor_info_t *recvTokensPerExpert,
                                     const zbccl_tensor_info_t *pushTargetOffset,
                                     const zbccl_tensor_info_t *balanceMatrix, zbccl_comm_t comm, aclrtStream stream,
                                     int64_t flags);

/**
 * @brief Calculate the layout required for communication
 *
 * @param topkIndex            [in] topK index info of per token
 * @param tokens               [in] num of tokens
 * @param expertNum            [in] num of experts
 * @param topkNum              [in] num of topK
 * @param tokensPerRank        [in/out] the number of tokens to be sent to each rank
 * @param tokensPerExpert      [in/out] the number of tokens to be sent to each expert
 * @param isTokenInRank        [in/out] whether a token be sent to a rank
 * @param sendTokensIndex      [in/out] send index of per token
 * @param notifySendData       [in/out] exchange data for notify
 * @param comm                 [in] zbccl communication handle
 * @param stream               [in] compute stream
 * @param flags                [in] optional flags, reserved or extend
 * @return
 */
int32_t zbccl_dispatch_normal_layout(const zbccl_tensor_info_t *topkIndex, int64_t tokens, int64_t expertNum,
                                     int64_t topkNum, const zbccl_tensor_info_t *tokensPerRank,
                                     const zbccl_tensor_info_t *tokensPerExpert,
                                     const zbccl_tensor_info_t *isTokenInRank, const zbccl_tensor_info_t *sendTokensIndex,
                                     const zbccl_tensor_info_t *notifySendData,
                                     zbccl_comm_t comm, aclrtStream stream, int64_t flags);

/**
 * @brief Dispatch operation in push mode
 *
 * @param srcTokens            [in] tensor info of source tokens to be dispatched
 * @param topkIndex            [in] topK index info of per token
 * @param sendTokensIndex      [in] send index of per token
 * @param pushTargetOffset     [in] the token offset sent by different ranks to each expert
 * @param expertNum            [in] number of export to be dispatch
 * @param quantMode            [in] quant mode
 * @param destTokens           [in/out] tensor info of destination tokens
 * @param destScale            [in/out] scale output after quant
 * @param comm                 [in] zbccl communication handle
 * @param stream               [in] stream
 * @param flags                [in] optional flags, reserved or extend
 * @return 0 if successful
 */
int32_t zbccl_dispatch_normal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *topkIndex,
                              const zbccl_tensor_info_t *sendTokensIndex, const zbccl_tensor_info_t *pushTargetOffset,
                              int64_t expertNum, zbccl_quant_mode_t quantMode, const zbccl_tensor_info_t *destTokens,
                              const zbccl_tensor_info_t *destScale, zbccl_comm_t comm, aclrtStream stream,
                              int64_t flags);

/**
 * @brief Combine operation in pull mode
 *
 * @param srcTokens            [in] tensor info of tokens be dispatched
 * @param srcTokensPerEp       [in] the number of tokens received by each expert from different ranks (prefix sum form)
 * @param topKWeight           [in] the weights of the topK experts for each token
 * @param topkIndex            [in] topK index info of per token
 * @param sendTokensIndex      [in] send index of per token
 * @param balanceMatrix        [in] the token range processed by each rank after balanced
 * @param expertNum            [in] moe expert number
 * @param destTokens           [in/out] tensor info of destination tokens
 * @param comm                 [in] zbccl communication handle
 * @param stream               [in] stream
 * @param flags                [in] optional flags, reserved or extend
 * @return
 */
int32_t zbccl_combine_normal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *srcTokensPerEp,
                             const zbccl_tensor_info_t *topKWeight, const zbccl_tensor_info_t *topkIndex,
                             const zbccl_tensor_info_t *sendTokensIndex, 
                             const zbccl_tensor_info_t *balanceMatrix, uint16_t expertNum,
                             const zbccl_tensor_info_t *destTokens, zbccl_comm_t comm, aclrtStream stream,
                             int64_t flags);

#ifdef __cplusplus
}
#endif

#endif  // ZBCCL_OPERATIONS_H_
