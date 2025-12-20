cmake_minimum_required(VERSION 3.16)
project(ascendc_op_tests)

set(SOC_VERSION "Ascend910_9382" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)

set(PROJECT_OP_SRC_BASE ${CMAKE_SOURCE_DIR}/csrc)
set(PROJECT_BUILD_PATH ${CMAKE_SOURCE_DIR}/build)
set(PROJECT_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/output)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

# ${KERNEL_FILES} are used to compile library, push files written by ascendc in ${KERNEL_FILES}.
# ref to cmake/npu.cmake ascendc_library, cmake/cpu.cmake add_library
file(GLOB KERNEL_FILES ${PROJECT_OP_SRC_BASE}/zccl/reduce_scatter/op_kernel/reduce_scatter_kernel.cpp)

# ascendc_library use to add kernel file to generate ascendc library
ascendc_library(ascendc_kernels_reduce_scatter STATIC ${KERNEL_FILES})
ascendc_include_directories(ascendc_kernels_reduce_scatter PRIVATE
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    ${ASCEND_INCLUDE_DIR}
)
# ascendc_compile_definitions(ascendc_kernels_reduce_scatter PRIVATE
#     ASCENDC_DUMP=0
# )

add_executable(ascendc_reduce_scatter ${CMAKE_CURRENT_SOURCE_DIR}/zccl/test_reduce_scatter.cpp)

target_compile_options(ascendc_reduce_scatter PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_include_directories(ascendc_reduce_scatter PRIVATE
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${ASCEND_INCLUDE_DIR}
)

target_link_directories(ascendc_reduce_scatter PRIVATE
    ${SHMEM_HOME_PATH}/shmem/lib/
    ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
    ${ASCEND_INCLUDE_DIR}
    ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(ascendc_reduce_scatter PRIVATE
    ascendc_kernels_reduce_scatter
    shmem
    mf_smem
    nnopbase
)

install(TARGETS ascendc_reduce_scatter
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
