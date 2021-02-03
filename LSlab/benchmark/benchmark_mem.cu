/*
 * Copyright (c) 2020-2021 dePaul Miller (dsm220@lehigh.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <StandardSlabDefinitions.cuh>
#include <vector>
#include <Slab.cuh>
#include <cuda_profiler_api.h>
#include <unordered_map>

template<>
struct EMPTY<int *> {
    static constexpr int *value = nullptr;
};

template<>
__forceinline__ __device__ unsigned compare(int *const &lhs, int *const &rhs) {
    return lhs - rhs;
}

int main() {

    const int size = 1000;
    std::hash<unsigned> hfn;
    SlabUnified<unsigned, int *> s(size);
    auto b = new BatchBuffer<unsigned, int *>();

    s.setGPU();

    for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
        unsigned j = 0;
        for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
            unsigned key = 1;
            int *value = new int[256]; // allocating 1KB
            for (int w = 0; w < 256; w++) {
                value[w] = 1;
            }
            b->getBatchKeys()[j] = key;
            b->getHashValues()[j] = hfn(key);
            b->getBatchRequests()[j] = REQUEST_INSERT;
            b->getBatchValues()[j] = value;
        }
        for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
            b->getBatchRequests()[j] = REQUEST_EMPTY;
        }
        s.moveBufferToGPU(b, 0x0);
        s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0);
        s.moveBufferToCPU(b, 0x0);
        gpuErrchk(cudaStreamSynchronize(0x0));
        j = 0;
        for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
            if (b->getBatchRequests()[j] == REQUEST_INSERT && b->getBatchValues()[j] != EMPTY<int *>::value) {
                delete[] b->getBatchValues()[j];
            }
        }
    }

    gpuErrchk(cudaProfilerStart());

    for (int rep = 0; rep < 10; rep++) {

        for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
                unsigned key = 1;
                int *value = new int[256]; // allocating 1KB
                for (int w = 0; w < 256; w++) {
                    value[w] = 1;
                }
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchRequests()[j] = REQUEST_INSERT;
                b->getBatchValues()[j] = value;
            }
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                b->getBatchRequests()[j] = REQUEST_EMPTY;
            }
            s.moveBufferToGPU(b, 0x0);
            s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0);
            s.moveBufferToCPU(b, 0x0);
            gpuErrchk(cudaStreamSynchronize(0x0));

            j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                if (b->getBatchRequests()[j] == REQUEST_INSERT && b->getBatchValues()[j] != EMPTY<int *>::value) {
                    delete[] b->getBatchValues()[j];
                }
            }
        }
    }

    gpuErrchk(cudaProfilerStop());
    delete b;
}