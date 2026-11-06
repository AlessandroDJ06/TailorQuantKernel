cmake_minimum_required(VERSION 3.20)
project(TailorQuantKernel C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Always optimize, also in CLion's default Debug profile.
set(CMAKE_C_FLAGS_DEBUG   "-O2 -g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(OpenMP REQUIRED)

add_library(tq STATIC
        packer/packer.c
        kernel/quant_x.c
        kernel/q8_scalar.c
        kernel/q4_scalar.c
        kernel/dispatch.c
)
target_include_directories(tq PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(tq PUBLIC OpenMP::OpenMP_C)
target_link_options(tq PUBLIC -fopenmp)
if(UNIX)
    target_link_libraries(tq PUBLIC m)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_sources(tq PRIVATE kernel/q8_avx2.c kernel/q4_avx2.c)
    set_source_files_properties(kernel/q8_avx2.c kernel/q4_avx2.c
            PROPERTIES COMPILE_OPTIONS "-mavx2")
    target_compile_definitions(tq PRIVATE TQ_HAVE_AVX2=1)
endif()

add_library(tq_model STATIC
        model/silu.c
        model/rmsnorm.c
        model/softmax.c
        model/rope.c
        model/attention.c
        model/attention_q8.c
        model/attention_layer.c
        model/kv_cache.c
        model/mlp.c
        model/embedding.c
        model/sampler.c
        model/vec.c
        model/context.c
)
target_link_libraries(tq_model PUBLIC tq)

add_executable(test_kernel tools/test_kernel.c)
target_link_libraries(test_kernel tq)

add_executable(test_model tools/test_model.c)
target_link_libraries(test_model tq_model)

add_executable(bench tools/bench.c)
target_link_libraries(bench tq)