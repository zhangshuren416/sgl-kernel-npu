import os
import torch
import torch_npu
import torch.distributed as dist
from typing import Callable, List, Optional, Tuple, Union

from zbccl.zbccl import deepep_adaptor
from zbccl.zbccl.deepep_adaptor import Config, EventHandle
from .zbccl_utils import EventOverlap


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using HCCS)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.(not used)
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode HCCS communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 12,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
    ) -> None:
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode HCCS communication. Use this name
                to ensure compatibility with DeepEP.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_low_latency_mode: This parameter is deprecated and retained to ensure compatibility with DeepEP.
            allow_mnnvl: This parameter is deprecated and retained to ensure compatibility with DeepEP.
        """

        self.rank = group.rank()
        self.group_size = group.size()

        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        try:
            backend = group._get_backend(torch.device("npu", self.rank))
            moe_all_to_all_group_name = backend.get_zbccl_comm_name()
        except Exception as e:
            print("get_hccl_comm_name failed", e)
            moe_all_to_all_group_name = ""
        self.runtime = deepep_adaptor.Buffer(
            self.rank,
            self.group_size,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode,
            moe_all_to_all_group_name,
        )

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.
        Argument:
            num_ranks: the number of ranks.
        Returns:
            config: the recommended config.
        """

        config_map = {
            2: Config(Buffer.num_sms, 24, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 36, 288, 20, 128),
            24: Config(Buffer.num_sms, 8, 288, 32, 128),
            32: Config(Buffer.num_sms, 32, 288, 32, 128),
            64: Config(Buffer.num_sms, 20, 288, 28, 128),
            128: Config(Buffer.num_sms, 20, 560, 32, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.
        Argument:
            num_ranks: the number of ranks.
        Returns:
            config: the recommended config.
        """

        config_map = {
            2: Config(Buffer.num_sms, 10, 256, 6, 128),
            4: Config(Buffer.num_sms, 9, 256, 6, 128),
            8: Config(Buffer.num_sms, 4, 256, 6, 128),
            16: Config(Buffer.num_sms, 4, 288, 12, 128),
            24: Config(Buffer.num_sms, 1, 288, 8, 128),
            32: Config(Buffer.num_sms, 1, 288, 8, 128),
            64: Config(Buffer.num_sms, 1, 288, 20, 128),
            128: Config(Buffer.num_sms, 1, 560, 12, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }

        return config_map[num_ranks]

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.
        Arguments:
            new_num_sms: the new number to be set.
        """
        assert new_num_sms % 2 == 0, "The SM count must be even"
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a NPU event on the current stream, i.e. `torch_npu.npu.current_stream()`.
        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        return deepep_adaptor.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap
    ]:
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.int`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = self.runtime.get_dispatch_layout(
            topk_idx,
            num_experts,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
        )
        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(event),
        )

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via HCCS.
        Internode kernels require the ranks in a node should be visible via HCCS, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be NPU-graph compatible. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.
            dispatch_wait_recv_cost_stats: `[num_ranks]` with `torch.int`, record the time it takes for the dispatch phase
                to receive all tokens from each slave rank in the current rank.

        Returns:
            recv_x: received tokens, the first element is a `torch.Tensor` shaped as `[received_token_count, hidden]` with
                `torch.int8`, the second tensor is the corresponding scales for the first element with shape `[received_token_count]`
                with `torch.float`.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            raise NotImplementedError("Not support internode")

        # Launch the kernel with cached or non-cached mode
        if isinstance(x, tuple):
            raise NotImplementedError("Not support fp8")
        x_scales = None
        use_quant = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT") == "1"

        if handle is not None:
            raise NotImplementedError(
                "Optional communication handle is not supported yet."
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )
            (
                recv_x,
                recv_x_scales,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                put_offset,
                balance_matrix,
                event,
            ) = self.runtime.intranode_dispatch(
                x,
                x_scales,
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                num_worst_tokens,
                config,
                getattr(previous_event, "event", None),
                async_finish,
                allocate_on_comm_stream,
                use_quant,
            )
            handle = (
                is_token_in_rank,
                topk_idx,
                topk_weights,
                put_offset,
                balance_matrix,
            )
            return (
                (recv_x, recv_x_scales) if use_quant else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(event),
            )

    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        combine_send_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via HCCS.
        Internode kernels require the ranks in a node should be visible via HCCS, while the ranks with the same NPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.
            combine_send_cost_stats: `[num_ranks]`: record the time when the current rank sends all tokens to other ranks
                in the combine phase.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            raise NotImplementedError("Not support internode")

        # NOTES: the second `_` is for the sending side, so we should use the third one
        (
            is_token_in_rank,
            topk_idx,
            topk_weights_ori,
            put_offset,
            balance_matrix,
        ) = handle

        # Launch the kernel
        recv_x, recv_topk_weights, event = self.runtime.intranode_combine(
            x,
            topk_idx,
            topk_weights_ori,
            put_offset,
            balance_matrix,
            getattr(previous_event, 'event', None),
            async_finish,
            allocate_on_comm_stream,
        )
        return recv_x, recv_topk_weights, EventOverlap(event)


    def clean_low_latency_buffer(
        self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int
    ) -> None:
        """
        As low-latency kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.
        For example, after running the normal dispatch/combine, you must run this function before executing any
            low-latency kernel.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        self.runtime.clean_low_latency_buffer(
            num_max_dispatch_tokens_per_rank, hidden, num_experts
        )