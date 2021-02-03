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

/**
 * @author dePaul Miller
 */
#include <atomic>
#include <future>
#include <mutex>
#include <thread>

#include "stdrequestHandler.cuh"

#ifndef MegaKVGPU_CUH
#define MegaKVGPU_CUH

namespace megakv {

    class MegaKVGPU {
    public:

        explicit MegaKVGPU(int size) {
            slab = setUp(size);
        }

        void batch(unsigned *keys, unsigned *values, int *requests, cudaStream_t stream, float &ms) {
            cudaEvent_t start, end;
            gpuErrchk(cudaEventCreate(&start));
            gpuErrchk(cudaEventCreate(&end));

            gpuErrchk(cudaEventRecord(start, stream));
            requestHandler<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(slab->slabs, slab->num_of_buckets, keys, values,
                                                                     requests);
            gpuErrchk(cudaEventRecord(end, stream));
            gpuErrchk(cudaEventSynchronize(end));
            gpuErrchk(cudaEventElapsedTime(&ms, start, end));
            gpuErrchk(cudaEventDestroy(start));
            gpuErrchk(cudaEventDestroy(end));
        }

        void exec_async(unsigned *keys, unsigned *values, int *requests, cudaStream_t stream) {
            requestHandler<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(slab->slabs, slab->num_of_buckets, keys, values,
                                                                     requests);
        }

    private:
        SlabCtx *slab;
    };

}
#endif // SLAB_SLAB_CUH
