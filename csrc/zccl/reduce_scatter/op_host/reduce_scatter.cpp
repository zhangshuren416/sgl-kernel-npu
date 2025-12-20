// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "defines.h"
#include "reduce_scatter_tilling.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_ShmemReduceScatter.h"
#include "torch_helper.h"
#include "shmem_api.h"

namespace sglang {
namespace npu_kernel {

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr uint32_t BIG_DATA_SIZE = 2 * 1024 * 1024;

HOST_API void zccl_reduce_scatter(const at::Tensor &tensor_a, at::Tensor &tensor_b)
{
    /* define the block dim */
    uint32_t blockDim = 8;

    /* memory size */
    uint32_t totalLength = 1;
    for (uint32_t size : tensor_a.sizes()) {
        totalLength *= size;
    }
    if (totalLength < BIG_DATA_SIZE) {
        blockDim = 8;
    } else {
        blockDim = 16;
    }

    /* launch the kernel function via torch */
    void *ptr = shmem_malloc(blockDim * SYNC_FLAG_INTERVAL * sizeof(int32_t) + GVA_BUFF_MAX_SIZE / sizeof(float));
    uint64_t fftsAddr = shmemx_get_ffts_config();
    uint32_t dataType = 0;
    uint32_t reduceOp = 0;
    EXEC_KERNEL_CMD(ShmemReduceScatter, blockDim, tensor_a, tensor_b, ptr, fftsAddr, dataType, totalLength, reduceOp);
    shmem_free(ptr);
}

}
}


