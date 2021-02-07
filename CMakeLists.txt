cmake_minimum_required(VERSION 3.14)
project(GPUHashMaps)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

enable_testing()

add_subdirectory(Allocator)
add_subdirectory(LSlab)
add_subdirectory(MegaKV)
add_subdirectory(docs)