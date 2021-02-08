# GPU Hashmaps Documentation

This CMake project contains the implementation of some 
GPU hashmaps. Currently the only documented parts are
the allocator and the LSlab hashmap. Note that this is
a header only library.

### Using Group Allocation

To use group allocation import GroupAllocator.cuh
or import groupallocator (which will import 
GroupAllocator.cuh). Check out both files for ways to
utilize them.

### Using LSlab

In order to use LSlab import Slab.cuh to start using.
It is best to look through the SlabUnified class to
get a sense of how to utilize the code.