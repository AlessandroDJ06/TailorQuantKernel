cmake_minimum_required(VERSION 3.20)
project(TailorQuantKernel C)

set(CMAKE_C_STANDARD 11)

# Zoek OpenMP
find_package(OpenMP REQUIRED)

add_library(tq STATIC
        packer/packer.c
        kernel/kernel.c
)
target_include_directories(tq PUBLIC ${CMAKE_SOURCE_DIR})
if(UNIX)
    target_link_libraries(tq PUBLIC m)
endif()

add_executable(test_kernel main.c)
target_link_libraries(test_kernel PRIVATE tq)

add_executable(bench bench.c)
target_link_libraries(bench PRIVATE tq)

target_compile_options(tq PRIVATE -O2)
target_compile_options(bench PRIVATE -O2)

if(OpenMP_C_FOUND)
    target_compile_options(tq PRIVATE ${OpenMP_C_FLAGS})
    target_link_options(tq PRIVATE ${OpenMP_C_FLAGS})

    target_compile_options(bench PRIVATE ${OpenMP_C_FLAGS})
    target_link_options(bench PRIVATE ${OpenMP_C_FLAGS})
endif()