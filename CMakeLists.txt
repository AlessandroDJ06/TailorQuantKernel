cmake_minimum_required(VERSION 4.3)
project(TailorQuantKernel C)

set(CMAKE_C_STANDARD 11)

add_executable(TailorQuantKernel main.c
        kernel/kernel.c
        kernel/kernel.h
        packer/packer.c
        packer/packer.h)
