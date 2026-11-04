cmake_minimum_required(VERSION 3.20)
project(TailorQuantKernel C)

set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_library(tq STATIC
        packer/packer.c
        kernel/kernel.c
)
target_include_directories(tq PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(tq PUBLIC OpenMP::OpenMP_C)
if(UNIX)
    target_link_libraries(tq PUBLIC m)
endif()

if(MSVC)
    target_compile_options(tq PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(tq PRIVATE -O2 -mavx2)
    target_compile_options(tq PUBLIC -fopenmp)
    target_link_options(tq PUBLIC -fopenmp)
endif()

add_executable(test_kernel main.c)
target_link_libraries(test_kernel PRIVATE tq)

add_executable(bench bench.c)
target_link_libraries(bench PRIVATE tq)
target_compile_options(bench PRIVATE $<$<NOT:$<C_COMPILER_ID:MSVC>>:-O2>)