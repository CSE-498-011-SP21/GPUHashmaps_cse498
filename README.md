# GPU Hashmaps

Implementations of our LSlab hashmap, MegaKV, and Slab.

## Compiling the Code

- Requires CMake version 3.14 at least
- Built using CUDA 11.2 
- Initialize the submodules and run the bootstrap script in vcpkg
- Install gtest with vcpkg
- Use the vcpkg/scripts/buildsystems/vcpkg.cmake file as 
  the CMAKE_TOOLCHAIN_FILE with cmake
- Compile with cmake for your system
- run make test for sanity checks