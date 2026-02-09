set(RUN_MODE "npu" CACHE STRING "run mode: npu")

ascendc_library(zbccl_kernel STATIC
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu/device/zbccl_kernel_allgather.cpp
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu/device/zbccl_kernel_allreduce.cpp
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu/device/zbccl_kernel_reduce_scatter.cpp
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu/device/zbccl_kernel_dispatch_layout.cpp
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu/device/zbccl_kernel_notify_dispatch.cpp
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu/device/zbccl_kernel_dispatch_normal.cpp
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu/device/zbccl_kernel_combine_normal.cpp
)

ascendc_include_directories(zbccl_kernel PRIVATE
        $ENV{SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/aclnn
        ${ASCEND_HOME_PATH}/include/aclnn/opdev
        ${ASCEND_HOME_PATH}/asc/include
        ${ZBCCL_SRC_DIR}/include
        ${ZBCCL_SRC_DIR}/csrc/common
        ${ZBCCL_SRC_DIR}/csrc/ccl
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu
)

ascendc_compile_definitions(zbccl_kernel PRIVATE
        ASCENDC_DUMP=0
)