import os
import torch
import torch_npu
import torch.distributed as dist
from typing import Callable, List, Optional, Tuple, Union

from zbccl.zbccl import deepep_adaptor
from zbccl.zbccl.deepep_adaptor import Config, EventHandle
from .zbccl_utils import EventOverlap


class Buffer:

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