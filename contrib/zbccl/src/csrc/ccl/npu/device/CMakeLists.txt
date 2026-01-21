ascendc_library(zbccl_kernel STATIC
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu/device/zbccl_kernel_allgather.cpp
)

ascendc_include_directories(zbccl_kernel PRIVATE
        $ENV{SHMEM_HOME_PATH}/shmem/include
        $ENV{SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/aclnn
        ${ASCEND_HOME_PATH}/include/aclnn/opdev
        ${ZBCCL_SRC_DIR}/include
        ${ZBCCL_SRC_DIR}/csrc/common
        ${ZBCCL_SRC_DIR}/csrc/ccl
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu
)